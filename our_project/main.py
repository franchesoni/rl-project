import os

import torch
from tqdm import trange

from cfg import CURRICULUM, _CURRICULUMS, CURRICULUM_SCHEDULE, N_INTERACTIONS, WRITER, MAX_DIGITS, TEACHER_NAME, SAVE_MODEL, SUMMARY_WRITER_PATH
from classroom import AdditionClassroom, CharacterTable, AdditionTask
from students import AdditionStudent, AdditionLSTM
from teachers import CurriculumTeacher, OnlineSlopeBanditTeacher, SamplingTeacher, RAWUCBTeacher


'''Welcome to the main script! The _main_ idea here (pun intended) is to 
create a classroom, assign a teacher (maybe with a handcrafted curriculum) and
a student, and then make some interactions.
Seed and writer dest are in classroom.py.'''

def run_specific_teacher_addition(teacher_name=TEACHER_NAME):
    if teacher_name == 'online':
        teacher = OnlineSlopeBanditTeacher(n_actions=MAX_DIGITS)
    elif teacher_name == 'curriculum':
        teacher = CurriculumTeacher(curriculum=CURRICULUM, curriculum_schedule=CURRICULUM_SCHEDULE, n_actions=len(CURRICULUM[0]))
    elif teacher_name == 'sampling':
        teacher = SamplingTeacher(n_actions=MAX_DIGITS)
    elif teacher_name == 'raw':   
        teacher = RAWUCBTeacher(n_actions=MAX_DIGITS)

    student = AdditionStudent()
    classroom = AdditionClassroom(teacher=teacher, student=student)
    for _ in trange(N_INTERACTIONS):
        classroom.step()

    if SAVE_MODEL:
        torch.save(student.model.state_dict(), os.path.join(SUMMARY_WRITER_PATH, "model.pt"))

def profile(function):  # I don't know where to put this
    '''Insights about time:
    - more than half of the time is currently located in
    get_observations (that is inside the training loop only for logging). This
    relative importance could be avoided if train_size is larger.
    - the processing time is similarly distributed between generate data and 
    the neural network processing.
    - teachers have little to no overhead.
    - config: {"N_INTERACTIONS": 10,"CURRICULUM": [[0, 0, 0, 1]], "MAX_DIGITS": 4, "TRAIN_SIZE": 100, "VAL_SIZE": 100, "BATCH_SIZE": 10, "EPOCHS": 10,
    '''
    import cProfile, pstats
    profiler = cProfile.Profile()
    profiler.enable()
    function()
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('cumtime')
    stats.print_stats(20)
    breakpoint()

def show_addition_examples(model_path, max_digits, n_examples):
    model = AdditionLSTM(max_digits=1)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    curriculum = _CURRICULUMS["direct"](max_digits)[0]
    add_task = AdditionTask(curriculum, 1000, curriculum, 1000, 1000, 1, max_digits)
    X, y, _ = add_task.generate_data(curriculum, n_examples)
    char_table = CharacterTable("0123456789+ ", 2*max_digits+1)
    x_pred = model(torch.from_numpy(X).float()).detach().numpy().transpose(1,0,2)
    print("question / prediction / solution")
    for i in range(n_examples):
        query = char_table.decode(X[i]).replace(" ", "_")
        sol = char_table.decode(y[i]).replace(" ", "_")
        pred = char_table.decode(x_pred[i]).replace(" ", "_")
        print("{} = {} ({})".format(query, pred, sol))

if __name__=='__main__':
    run_specific_teacher_addition()


    # profile(run_specific_teacher_addition)

    # # the following now works
    # from teachers import test_RAWUCB
    # test_RAWUCB()
