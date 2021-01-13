import warnings

import numpy as np
import torch

from cfg import (BATCH_SIZE, EPOCHS, MAX_DIGITS, NUM_CHARS, TRAIN_SIZE,
    VAL_SIZE, WRITER)


"""This is the longest and ugliest code. Have a look at <AbstractClassroom>
to get an idea of the utility of having a classroom: it basically generates the
tasks signaled by the teacher and passes them to the student, then observes the
student and comes back to the teacher with a reward signal."""


class AbstractClassroom:
    def __init__(self, teacher, student):
        self.teacher = teacher
        self.student = student
        self.reward = None
        self.global_step = 0

    def generate_task(self, task_index):
        """Generates a new task according to index"""
        raise NotImplementedError(
            "This is an abstract class, you should implement this method"
        )

    def compute_reward(self, obs):
        """Computes the reward from obs"""
        raise NotImplementedError(
            "This is an abstract class, you should implement this method"
        )

    def step(self):
        self.global_step += 1
        task_dist = self.teacher.give_task(self.reward)
        task = self.generate_task(task_dist)
        obs = self.student.learn_from_task(task)
        self.reward = self.compute_reward(obs)

        WRITER.add_scalars("Classroom/observations", {str(i+1):ob for i, ob in enumerate(obs)}, self.global_step)
        WRITER.add_scalars("Classroom/rewards", {str(i+1):r for i, r in enumerate(self.reward)}, self.global_step)
        WRITER.add_scalars("Classroom/distribution", {str(i+1):d for i, d in enumerate(task_dist)}, self.global_step)


"""The abstract task has the training parameters for the student learning. It
uses probability distributions over the tasks. This allows the teacher to give
only one task (by setting that task as with probability 1) or mix tasks. The
mixing is inside the generating data. We should see how much efficiency is lost
by reducing batch size, as it's needed for the bandit framework. Another option
is to set tasks as being a set of handcrafted curricula."""


class AbstractTask:
    def __init__(
        self, train_dist, train_size, val_dist, val_size, batch_size=4096, epochs=1,
    ):
        self.train_dist = train_dist
        self.train_size = train_size
        self.val_dist = val_dist
        self.val_size = val_size
        self.batch_size = batch_size
        self.epochs = epochs

    def generate_data(self, dist, size):
        """Generates X and y arrays of examples of size=size drawing examples
        from pdf dist"""
        raise NotImplementedError(
            "This is an abstract class, you should implement this method"
        )

    def loss_fn(self, pred, y):
        raise NotImplementedError(
            "This is an abstract class, you should implement this method"
        )

    def get_observation(self, model, val_size=VAL_SIZE):
        """Computes the observation given model. This should be reimplemented
        inside Student if it's too inefficient to make a whole new computation."""
        raise NotImplementedError(
            "This is an abstract class, you should implement this method"
        )

    def val_score_fn(self, val_pred, val_y, val_lens):
        raise NotImplementedError(
            "This is an abstract class, you should implement this method"
        )

    def finished(self, val_score):
        """Returns bool telling if val score is enough to finish. It assumes
        the finishing criterium is based on val score. It also finishes when
        n_epochs is completed"""
        raise NotImplementedError(
            "This is an abstract class, you should implement this method"
        )


########################################################################################
########################################################################################
########################################################################################

# TOY ROTTING BANDIT

###############################################
"""This does not inherit from abstract classroom because it's not a teacher-stu
dent framework. This toy problem comes from the original rotting bandit paper:
"use Normal distributions with σ 2 = 0.2, and T = 30, 000.
Non-Parametric: K = 2. As for the expected rewards: μ 1 (n) = 0.5, ∀n, and μ 2 (n) = 1 for its first
7, 500 pulls and 0.4 afterwards:
"""


class ToyRottingProblem:
    def __init__(self):
        self.K = 2
        self.sigma = np.sqrt(0.2)
        self.T = 30000
        self.t = 0

    def reset(self):
        self.__init__()

    def step(self, chosen_arm):
        assert chosen_arm in [0, 1]
        zero_centered_reward = np.random.randn(1) * self.sigma
        if chosen_arm == 0:
            reward = zero_centered_reward + 0.5
        elif chosen_arm == 1 and self.t < 7500:
            reward = zero_centered_reward + 1
        else:
            reward = zero_centered_reward + 0.4
        self.t += 1
        if not self.t < 1 + self.T:
            print("Problem has ended, resetting")
            self.reset()
            return None
        return np.ones(2) * reward


########################################################################################
########################################################################################
########################################################################################

# ADDITION

###############################################
class AdditionClassroom(AbstractClassroom):
    def __init__(self, teacher, student):
        super().__init__(teacher, student)
        self.past_obs = [0]  # initialize with something so we can take diff

    def generate_task(self, task_dist):
        """Generates a new task according to task_dist. Task dist should be
        a one hot vector when all work is finished."""
        val_dist = np.zeros_like(task_dist)  # validation distribution is on final task by default
        val_dist[-1] = 1  # final task
        task = AdditionTask(
            train_dist=task_dist,
            train_size=TRAIN_SIZE,
            val_dist=val_dist,
            val_size=VAL_SIZE,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            max_digits=MAX_DIGITS,
        )
        return task

    def compute_reward_diff(self, obs, warn=False):
        self.past_obs.append(obs)
        if warn:
            warnings.warn(RuntimeWarning('This is growing memory'))
        return self.past_obs[-1] - self.past_obs[-2]

    def compute_reward(self, obs):
        return self.compute_reward_diff(obs)


class AdditionClassroom3(AdditionClassroom):
    def compute_reward(self, obs):
        return np.abs(self.compute_reward_diff(obs))

class AdditionClassroom2(AdditionClassroom):
    def generate_task(self, task_dist):
        """Generates a new task according to task_dist. Task dist should be
        a one hot vector when all work is finished."""
        val_dist = np.zeros_like(task_dist)  # validation distribution is on final task by default
        val_dist[-1] = 1  # final task
        task = AdditionTask2(
            train_dist=task_dist,
            train_size=TRAIN_SIZE,
            val_dist=val_dist,
            val_size=VAL_SIZE,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            max_digits=MAX_DIGITS,
        )
        return task

    def compute_reward(self, obs):
        return - np.array([obs] * len(MAX_DIGITS))  # multidigit loss

'''This is the task that has _everything_ (except for the NN model, in student,
and the decision policy, in the teacher, and the reward computation, in the
classroom)'''

class AdditionTask(AbstractTask):
    def __init__(
        self,
        train_dist,
        train_size,
        val_dist,
        val_size,
        batch_size=4096,
        epochs=1,
        max_digits=15,
        invert=True,
    ):
        super().__init__(
            train_dist, train_size, val_dist, val_size, batch_size, epochs,
        )
        self.val_dist = val_dist
        self.uniform_dist = np.ones_like(val_dist) / len(val_dist)
        self.max_digits = max_digits
        self.invert = invert
        self.maxlen = max_digits + 1 + max_digits
        self.chars = "0123456789+ "
        self.num_chars = len(self.chars)
        assert self.num_chars == NUM_CHARS  # check consistency between scripts
        self.ctable = CharacterTable(self.chars, self.maxlen)

    def shuffle_along_axis(self, a, axis):
        # from https://stackoverflow.com/a/55317373/8462678
        idx = np.random.rand(*a.shape).argsort(axis=axis)
        return np.take_along_axis(a,idx,axis=axis)


    def generate_data(self, dist, size):
        """Generates onehot encoded batch X of shape (size, maxlen, num_chars)
        and y of shape (size, max_digits+1, num_chars). The order of the one hot
        encoding is reversed: "2021+20  " is passed as the onehot encoded version
        of "  02+1202". The answer is " 2041" and is encoded without inversion."""
        questions = []
        expected = []
        lengths = []
        gen_digitss = 1 + np.random.choice(len(dist),size=size,p=dist)
        numbers = np.random.choice(list('0123456789'), size=sum(gen_digitss)*2)
        start_ind = 0
        for i, gen_digits in enumerate(gen_digitss):
            a = "".join(numbers[start_ind:start_ind+gen_digits])
            b = "".join(numbers[start_ind+gen_digits:start_ind+2*gen_digits])
            a, b = int(a), int(b)  # as in the paper  # BREAKING CHANGE
            start_ind += gen_digits*2
            # Pad the data with spaces such that it is always MAXLEN
            query = "{}+{}".format(a, b).ljust(self.maxlen)
            ans = str(int(a) + int(b))
            # Answers can be of maximum size DIGITS + 1
            ans = ans.ljust(self.max_digits + 1)
            if self.invert:
                query = query[::-1]  # spaces first. It could have zeros first too.
            questions.append(query)
            expected.append(ans)
            lengths.append(gen_digits)

        X = np.zeros((len(questions), self.maxlen, self.num_chars), dtype=np.bool,)
        y = np.zeros(
            (len(questions), self.max_digits + 1, self.num_chars), dtype=np.bool,
        )
        for i, sentence in enumerate(questions):
            X[i] = self.ctable.encode(sentence, maxlen=self.maxlen)
        for i, sentence in enumerate(expected):
            y[i] = self.ctable.encode(sentence, maxlen=self.max_digits + 1)
        return X, y, np.array(lengths)

    def accuracy_per_length(self, y_pred, y, lengths, warn=False):
        """Computes accuracy using model output """
        y = np.argmax(y, axis=-1)  # target indices  (batch, max_digits+1)
        p = np.argmax(y_pred, axis=-1)  # inferred indices  (batch, max_digits+1)
        accs = []
        for i in range(self.max_digits):
            yl = y[lengths == i + 1]  # select those labels with length
            pl = p[lengths == i + 1]
            tf = np.all(yl == pl, axis=1)  # set to true those that coincide
            accs.append(np.mean(tf))
        if warn:
            warnings.warn(
                "accuracy per length computed without pytorch (using numpy only)",
                RuntimeWarning,
            )
        return np.array(accs)

    def full_number_accuracy(self, y_pred, y_true):
        """Accuracy in the sense of y_true and y_pred being identical"""
        y_true_argmax = torch.argmax(y_true, dim=-1)
        y_pred_argmax = torch.argmax(y_pred, dim=-1)
        tfd = torch.eq(y_true_argmax, y_pred_argmax)
        tfn = torch.all(tfd, dim=1)
        tfc = tfn.float()
        return torch.mean(tfc)

    def loss_fn(self, y_pred, y_true, lengths):
        '''Categorical crossentropy loss'''
        # dimensions as in https://stackoverflow.com/questions/60121107/pytorch-nllloss-function-target-shape-mismatch
        y_true_argmax = torch.argmax(y_true, dim=-1).view(-1)  # flatten
        return torch.nn.NLLLoss()(
            torch.log(y_pred).reshape(-1, y_pred.shape[2]), y_true_argmax
        )

    def get_observation(self, model, val_size=VAL_SIZE):
        """Computes the observation given model. This should be reimplemented
        inside Student if it's too inefficient to make a whole new computation."""
        val_data = self.generate_data(self.uniform_dist, val_size)
        val_X, val_y, val_lens = val_data
        val_X = torch.from_numpy(val_X).float().to(model.device)
        pred = model(val_X).transpose(0, 1)
        return self.accuracy_per_length(pred.cpu().detach().numpy(), val_y, val_lens)

    def val_score_fn(self, val_pred, val_y, val_lens):
        return self.accuracy_per_length(val_pred.cpu().detach().numpy(), val_y, val_lens)
        # return self.full_number_accuracy(val_pred, val_y)

    def finished(self, val_score):
        """Returns bool telling if val score is enough to finish. It assumes
        the finishing criterium is based on val score. It also finishes when
        n_epochs is completed"""
        return val_score > 0.99



class CharacterTable(object):
    """
    Given a set of characters:
    + Encode them to a one hot integer representation
    + Decode the one hot integer representation to their character output
    + Decode a vector of probabilities to their character output
    """

    def __init__(self, chars, maxlen):
        """ one hot encoded vector dim: (maxlen, #chars)
        """
        self.maxlen = maxlen
        # sorted list of chars without duplicates
        # self.chars = sorted(set(chars))
        self.chars = chars
        # access dict with char to find index as x_ind = self.char_indices['x']
        self.char_to_indices = {c: i for i, c in enumerate(self.chars)}
        # access dict with index to find char. It should be the same than self.chars[ind]
        self.indices_to_char = {i: c for i, c in enumerate(self.chars)}

    def encode(self, C, maxlen=None):
        """encode string C as one-hot encoding
        """
        assert type(C) == str  # check type
        maxlen = maxlen or self.maxlen  # overwrite maxlen if passed
        assert len(C) <= maxlen, "can not encode a sequence larger than maxlen"
        X = np.zeros(
            (maxlen, len(self.chars))
        )  # one hot encoding dim: (maxlen, #chars)
        for i, c in enumerate(C):
            # set a 1 in ith row and corresponding column
            X[i, self.char_to_indices[c]] = 1
        return X

    def decode(self, X):
        X = X.argmax(axis=-1)  # gets first index of greatest integer
        return "".join(self.indices_to_char[x] for x in X)


class AdditionTask2(AdditionTask):
    def get_observation(self, model, val_size=VAL_SIZE):
        """Computes the observation given model. This should be reimplemented
        inside Student if it's too inefficient to make a whole new computation."""
        val_data = self.generate_data(self.uniform_dist, val_size)
        val_X, val_y, val_lens = val_data
        model.eval()
        with torch.no_grad():
            val_X = torch.from_numpy(val_X).float().to(model.device)
            y_pred = model(val_X).transpose(0, 1)
            y_true_argmax = torch.argmax(val_y, dim=-1).view(-1)  # flatten
            avgloss = torch.nn.NLLLoss()(
                torch.log(y_pred).reshape(-1, y_pred.shape[2]), y_true_argmax
            )
        return avgloss

