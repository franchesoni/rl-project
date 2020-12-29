from cfg import CURRICULUM, N_INTERACTIONS, WRITER
from classroom import TestAdditionClassroom
from students import AdditionStudent
from teachers import CurriculumTeacher


'''Welcome to the main script! The _main_ idea here (pun intended) is to 
create a classroom, assign a teacher (maybe with a handcrafted curriculum) and
a student, and then make some interactions.
Seed and writer dest are in classroom.py.'''


if __name__=='__main__':

    WRITER.add_text('Progress', 'Started script')

    teacher = CurriculumTeacher(curriculum=CURRICULUM, n_actions=len(CURRICULUM[0]))
    student = AdditionStudent()
    classroom = TestAdditionClassroom(teacher=teacher, student=student)

    for _ in range(N_INTERACTIONS):
        classroom.step()
    