
import torch
from cfg import CURRICULUM, N_INTERACTIONS, WRITER, MAX_DIGITS, TEACHER_NAME, SAVE_MODEL, SUMMARY_WRITER_PATH
from classroom import AdditionClassroom
from students import AdditionStudent
from teachers import CurriculumTeacher, OnlineSlopeBanditTeacher, SamplingTeacher, RAWUCBTeacher


'''Welcome to the main script! The _main_ idea here (pun intended) is to 
create a classroom, assign a teacher (maybe with a handcrafted curriculum) and
a student, and then make some interactions.
Seed and writer dest are in classroom.py.'''

def run_specific_teacher_addition(teacher_name=TEACHER_NAME):
    if teacher_name == 'online':
        teacher = OnlineSlopeBanditTeacher(n_actions=MAX_DIGITS)
    elif teacher_name == 'curriculum':
        teacher = CurriculumTeacher(curriculum=CURRICULUM, n_actions=len(CURRICULUM[0]))
    elif teacher_name == 'sampling':
        teacher = SamplingTeacher(n_actions=MAX_DIGITS)
    elif teacher_name == 'raw':   
        teacher = RAWUCBTeacher(n_actions=MAX_DIGITS)

    student = AdditionStudent()
    classroom = AdditionClassroom(teacher=teacher, student=student)
    for _ in range(N_INTERACTIONS):
        # breakpoint()
        classroom.step()

    if SAVE_MODEL:
        torch.save(student.model.state_dict(), SUMMARY_WRITER_PATH)
if __name__=='__main__':

    import cProfile, pstats
    profiler = cProfile.Profile()
    profiler.enable()
    run_specific_teacher_addition()
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('cumsum')
    stats.print_stats()



    # # the following now works
    # from teachers import test_RAWUCB
    # test_RAWUCB()
