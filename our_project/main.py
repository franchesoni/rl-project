import os

import numpy as np
import torch
from tqdm import trange

from cfg import (CONFIG_FILE, CURRICULUM, _CURRICULUMS, CURRICULUM_SCHEDULE,
    N_INTERACTIONS, WRITER, MAX_DIGITS, TEACHER_NAME, SAVE_MODEL, SHOW_ADD,
    ABSOLUTE, SUMMARY_WRITER_PATH, CLASS_NUMBER, BOLTZMANN_TEMPERATURE)
from classroom import AdditionClassroom, AdditionClassroom2, AdditionClassroom3, CharacterTable, AdditionTask
from students import AdditionStudent, AdditionLSTM
from teachers import (CurriculumTeacher, OnlineSlopeBanditTeacher,
    SamplingTeacher, RAWUCBTeacher)


'''Welcome to the main script! The _main_ idea here (pun intended) is to 
create a classroom, assign a teacher (maybe with a handcrafted curriculum) and
a student, and then make some interactions.
Seed and writer dest are in classroom.py.'''



def run_specific_teacher_addition(
        teacher_name=TEACHER_NAME, show_addition=SHOW_ADD,
        show_freq=10, dist_show="direct"):
    if teacher_name == 'online':
        teacher = OnlineSlopeBanditTeacher(
            n_actions=MAX_DIGITS, absolute=ABSOLUTE,
            temperature=BOLTZMANN_TEMPERATURE)
    elif teacher_name == 'curriculum':
        teacher = CurriculumTeacher(
            curriculum=CURRICULUM, curriculum_schedule=CURRICULUM_SCHEDULE,
            n_actions=len(CURRICULUM[0]))
    elif teacher_name == 'sampling':
        teacher = SamplingTeacher(
            n_actions=MAX_DIGITS, absolute=ABSOLUTE)
    elif teacher_name == 'raw':   
        teacher = RAWUCBTeacher(
            n_actions=MAX_DIGITS)
    else:
        raise ValueError

    student = AdditionStudent()
    if CLASS_NUMBER == 1:
        classroom = AdditionClassroom(teacher=teacher, student=student)
    elif CLASS_NUMBER == 2:
        classroom = AdditionClassroom2(teacher=teacher, student=student)
    elif CLASS_NUMBER == 3:
        classroom = AdditionClassroom3(teacher=teacher, student=student)
    else:
        raise ValueError("CLASS_NUMBER {} is not a valid classroom number.".format(CLASS_NUMBER))
    
    pbar = trange(N_INTERACTIONS)
    for i in pbar:
        pbar.set_description("Processing {}".format(CONFIG_FILE))
        classroom.step()
        if i % show_freq == 0 and show_addition:
            model_path = os.path.join(
                SUMMARY_WRITER_PATH, "model_{}.pt".format(i))
            torch.save(student.model.state_dict(), model_path)
            show_addition_examples(
                model_path, MAX_DIGITS, n_examples=100, nb_print=10, dist=dist_show)

    if SAVE_MODEL:
        torch.save(student.model.state_dict(),
            os.path.join(SUMMARY_WRITER_PATH, "model.pt"))

def profile(function):  # I don't know where to put this
    '''Insights about time:
    - more than half of the time is currently located in
    get_observations (that is inside the training loop only for logging). This
    relative importance could be avoided if train_size is larger.
    - the processing time is similarly distributed between generate data and 
    the neural network processing.
    - teachers have little to no overhead.
    - config: {"N_INTERACTIONS": 10,"CURRICULUM": [[0, 0, 0, 1]], "MAX_DIGITS": 4,
        "TRAIN_SIZE": 100, "VAL_SIZE": 100, "BATCH_SIZE": 10, "EPOCHS": 10,
    '''
    import cProfile, pstats
    profiler = cProfile.Profile()
    profiler.enable()
    function()
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('cumtime')
    stats.print_stats(20)
    breakpoint()

examples_count = 0
def show_addition_examples(
        model_path, max_digits, n_examples=5, nb_print=5, dist="direct", only_wrong=True):
    global examples_count
    examples_count += 1
    model = AdditionLSTM(max_digits=max_digits)
    model.load_state_dict(torch.load(model_path))
    os.remove(model_path)
    model.eval()
    if dist == "direct":
        curriculum = _CURRICULUMS["direct"](max_digits)[0]
    elif dist == "uniform":
        curriculum = _CURRICULUMS["baseline"](max_digits)[0]
    else:
        raise ValueError("dist {} not in ['direct', 'uniform'].".format(dist))
    add_task = AdditionTask(
        curriculum, 1000, curriculum, 1000, 1000, 1, max_digits)
    X, y, lengths = add_task.generate_data(curriculum, n_examples)
    char_table = CharacterTable("0123456789+ ", 2*max_digits+1)
    y = torch.from_numpy(y).flip(dims=[1]).detach().numpy()
    y_pred = model(torch.from_numpy(X).float()).detach().numpy().transpose(1,0,2)
    y_pred = torch.from_numpy(y_pred).flip(dims=[1]).detach().numpy()
    if only_wrong:
        i, i_printed = 0, 0
        y2 = np.argmax(y, axis=-1)  # target indices  (batch, max_digits+1)
        p = np.argmax(y_pred, axis=-1)  # inferred indices  (batch, max_digits+1)
        while (i < n_examples) and (i_printed < nb_print):
            y3 = y2[i]
            p3 = p[i]
            if np.any(y3 != p3):
                query = char_table.decode(X[i])[::-1]
                pred = char_table.decode(y_pred[i])[::-1]
                sol = char_table.decode(y[i])[::-1]
                WRITER.add_text('examples', "'{}' = '{}' ('{}')\n".format(query, pred, sol), examples_count)
                WRITER.add_text('accuracy', str(add_task.accuracy_per_length(y_pred, y, lengths)), examples_count)
                i_printed += 1
            i += 1
    else:
        for i in range(min(nb_print, n_examples)):
            query = char_table.decode(X[i])[::-1]
            pred = char_table.decode(y_pred[i])[::-1]
            sol = char_table.decode(y[i])[::-1]
            print("'{}' = '{}' ('{}')".format(query, pred, sol))
            print(">>> Accuracy: {}\n".format(
                add_task.accuracy_per_length(y_pred, y, lengths)))
    return X, y, lengths, model

if __name__=='__main__':

    run_specific_teacher_addition(dist_show="uniform")
