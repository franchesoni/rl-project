# Teacher guided curriculum learning as a rotting bandit problem
Here lies the project structure.

The important configuration parameters for doing custom experiments are in any `config*.json` file.

## Description of the scripts

### main.py
This script creates a student, a teacher (decided in config) and a classroom, and every `show_freq=10` steps (not in config), shows examples of sums computed by the model. It also allows profiling to see what is taking the most time.

### classroom.py
Here is where the actions happens. This script implements the `AbstractClassroom` class, that takes a task distribution (it's more general than a task) given by the teacher and creates a task accordingly. This task is given to the student to learn, who returns an observation, from which the classroom computes the reward signal, to be given to the teacher.
The script also implement the `Task` class, which can `generate_data`, `get_observation`, say wheather the student has `finished`, and compute `val_score_fn` and `loss_fn`.

All of those methods (and more) are defined for the `AdditionClassroom`.

### students.py
Here lies the implementation of `AbstractStudent`, that can `learn_from_task` when given a `Task`, using traditional Pytorch training loops. The main role of the student is to have the training loops and the model. For the addition example, we implement a custom model.

### teachers.py
In this script are implemented the different teachers, which can `give_task` (and collect the last reward at the same time). This are mainly some type of bandit algorithms.



