import warnings

import numpy as np
import torch

from cfg import (
    BATCH_SIZE,
    EPOCHS,
    MAX_DIGITS,
    NUM_CHARS,
    TRAIN_SIZE,
    OBS_SIZE,
    WRITER,
    REWARD_FN,
    OBS_TYPE,
    MODE,
)

"""This is the longest and ugliest code. Have a look at <AbstractClassroom>
to get an idea of the utility of having a classroom: it basically generates the
tasks signaled by the teacher and passes them to the student, then observes the
student and comes back to the teacher with a reward signal."""


class AbstractClassroom:
    def __init__(self, teacher, student):
        self.teacher = teacher
        self.student = student
        self.reward = None
        self.class_step = 0

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
        self.class_step += 1
        task_dist = self.teacher.give_task(self.reward)
        task = self.generate_task(task_dist)
        obs = self.student.learn_from_task(task)
        self.reward = self.compute_reward(obs)

        if self.student.update_count % self.student.logger.freq == 0:
            WRITER.add_scalars("Classroom/observations", {str(i+1):ob for i, ob in enumerate(obs)}, self.student.update_count)
            WRITER.add_scalars("Classroom/rewards", {str(i+1):r for i, r in enumerate(self.reward)}, self.student.update_count)
            WRITER.add_scalars("Classroom/distribution", {str(i+1):d for i, d in enumerate(task_dist)}, self.student.update_count)


"""The abstract task has the training parameters for the student learning. It
uses probability distributions over the tasks. This allows the teacher to give
only one task (by setting that task as with probability 1) or mix tasks. The
mixing is inside the generating data. We should see how much efficiency is lost
by reducing batch size, as it's needed for the bandit framework. Another option
is to set tasks as being a set of handcrafted curricula."""


class AbstractTask:
    def __init__(
        self, train_dist, train_size, log_dict, obs_size, batch_size=4096, epochs=1,
    ):
        self.train_dist = train_dist
        self.train_size = train_size
        self.log_dict = log_dict
        self.obs_size = obs_size
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

    def get_observation(self, model, size=OBS_SIZE):
        """Computes the observation given model. This should be reimplemented
        inside Student if it's too inefficient to make a whole new computation."""
        raise NotImplementedError(
            "This is an abstract class, you should implement this method"
        )

    def logging_fn(self, *args):
        raise NotImplementedError(
            "This is an abstract class, you should implement this method"
        )

    def finished(self, score):
        """Returns bool telling if val score is enough to finish. It assumes
        the finishing criterium is based on val score. It also finishes when
        n_epochs is completed"""
        raise NotImplementedError(
            "This is an abstract class, you should implement this method"
        )


########################################################################################
########################################################################################
########################################################################################

# ADDITION

###############################################################################
class AdditionClassroom(AbstractClassroom):
    def __init__(self, teacher, student, reward_fn=REWARD_FN, mode=MODE):
        super().__init__(teacher, student)
        # aux variables
        self.set_reward_fn(reward_fn)
        self.set_task(mode)

    def set_reward_fn(self, reward_fn):
        if reward_fn == "absolute":
            self.reward_fn = np.abs
        elif reward_fn == "square":
            self.reward_fn = np.square
        elif reward_fn == "identity":
            self.reward_fn = lambda x: x
        elif reward_fn == "neg identity":
            self.reward_fn = lambda x: -x
        elif reward_fn == "square root":
            self.reward_fn = lambda x: np.sqrt(np.abs(x))
        elif reward_fn == "diff":
            self.past_obs = None  # initialize with something so we can take diff
            self.reward_fn = self._compute_reward_diff
        elif reward_fn == "rel diff":
            self.past_obs = None  # initialize with something so we can take diff
            self.reward_fn = self._compute_reward_rel_diff
        # one could concatenate things here
        else:
            raise ValueError("reward_fn: {} not valid".format(reward_fn))

    def _compute_reward_rel_diff(self, obs):
        if self.past_obs is None:
            self.past_obs = [obs]
        self.past_obs.append(obs)
        try:
            return (self.past_obs[-1] - self.past_obs[-2]) / self.past_obs[-2]
        except ZeroDivisionError:
            return self.past_obs[-2]

    def _compute_reward_diff(self, obs, warn=False):
        if self.past_obs is None:
            self.past_obs = [obs]
        self.past_obs.append(obs)
        return self.past_obs[-1] - self.past_obs[-2]


    def set_task(self, task):
        if task == "bandit":
            self.generate_task = self._generate_bandit_task
        elif task == "sequential":
            self.generate_task = self._generate_sequential_task

    def _generate_sequential_task(self, task_dist):
        """Generates a new task according to task_dist. Task dist should be
        a one hot vector when all work is finished."""
        task = AdditionSequentialTask(
            train_dist=task_dist,
            train_size=TRAIN_SIZE,
            obs_size=OBS_SIZE,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            max_digits=MAX_DIGITS,
        )
        return task

    def _generate_bandit_task(self, task_dist):
        """Generates a new task according to task_dist. Task dist should be
        a one hot vector when all work is finished."""
        task = AdditionBanditTask(
            train_dist=task_dist,
            train_size=TRAIN_SIZE,
            obs_size=OBS_SIZE,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            max_digits=MAX_DIGITS,
        )
        return task

    def compute_reward(self, obs):
        # do something
        return self.reward_fn(obs)


"""This is the task that has _everything_ (except for the NN model, in student,
and the decision policy, in the teacher, and the reward computation, in the
classroom)"""


class AdditionTask(AbstractTask):
    def __init__(
        self,
        train_dist,
        train_size,
        obs_size,
        batch_size,
        epochs,
        max_digits,
        invert=True,
    ):
        super().__init__(
            train_dist, train_size, obs_size, batch_size, epochs,
        )
        self.max_digits = max_digits
        self.invert = invert
        self.maxlen = max_digits + 1 + max_digits
        self.chars = "0123456789+ "
        self.num_chars = len(self.chars)
        assert self.num_chars == NUM_CHARS  # check consistency between scripts
        self.ctable = CharacterTable(self.chars, self.maxlen)
        self.last_loss = None  # to be overwrited by student

    def generate_data(self, dist, size):
        """Generates onehot encoded batch X of shape (size, maxlen, num_chars)
        and y of shape (size, max_digits+1, num_chars). The order of the one hot
        encoding is reversed: "2021+20  " is passed as the onehot encoded version
        of "  02+1202". The answer is " 2041" and is encoded without inversion."""
        questions = []
        expected = []
        lengths = []
        gen_digitss = 1 + np.random.choice(len(dist), size=size, p=dist)
        numbers = np.random.choice(list("0123456789"), size=sum(gen_digitss) * 2)
        start_ind = 0
        for i, gen_digits in enumerate(gen_digitss):
            a = "".join(numbers[start_ind : start_ind + gen_digits])
            b = "".join(numbers[start_ind + gen_digits : start_ind + 2 * gen_digits])
            a, b = int(a), int(b)  # as in the paper  # BREAKING CHANGE
            start_ind += gen_digits * 2
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

    # def val_score_fn(self, val_pred, val_y, val_lens):
    #     return self.accuracy_per_length(val_pred.cpu().detach().numpy(), val_y, val_lens)
    #     # return self.full_number_accuracy(val_pred, val_y)

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

    def per_digit_loss(self, y_pred, obs_y, obs_lens, device="cpu"):
        y_true_argmax = torch.from_numpy(np.argmax(obs_y, axis=-1)).to(device)
        obs_loss = np.array(
            [
                torch.nn.NLLLoss()(
                    torch.log(y_pred[obs_lens == i + 1]).transpose(1, 2),
                    y_true_argmax[obs_lens == i + 1],
                ).item()
                for i in range(self.max_digits)
            ]
        )
        return obs_loss

    def loss_fn(self, y_pred, y_true, lengths):
        """Categorical crossentropy loss"""
        # dimensions as in https://stackoverflow.com/questions/60121107/pytorch-nllloss-function-target-shape-mismatch
        y_true_argmax = torch.argmax(y_true, dim=-1).view(-1)  # flatten
        return torch.nn.NLLLoss()(
            torch.log(y_pred).reshape(-1, y_pred.shape[2]), y_true_argmax
        )

    def finished(self, val_score):
        """Returns bool telling if val score is enough to finish. It assumes
        the finishing criterium is based on val score. It also finishes when
        n_epochs is completed"""
        return val_score > 0.99


class AdditionSequentialTask(AdditionTask):
    def __init__(self, *args, obs_type=OBS_TYPE, **kwargs):
        super().__init__(*args, **kwargs)
        self.set_get_observation(obs_type)

    def set_get_observation(self, obs_type):
        if obs_type == "per_digit_loss":
            self.get_observation = self.get_observation_per_digit_loss
        elif obs_type == "accuracy_per_length":
            self.get_observation = self.get_observation_accuracy_per_length
        else:
            raise ValueError('obs_type is incorrect')

    def get_observation_accuracy_per_length(self, model, size=OBS_SIZE):
        """Computes the observation given model. This should be reimplemented
        inside Student if it's too inefficient to make a whole new computation."""
        obs_data = self.generate_data(self.max_digits * [1 / self.max_digits], size)
        obs_X, obs_y, obs_lens = obs_data
        obs_X = torch.from_numpy(obs_X).float().to(model.device)
        pred = model(obs_X).transpose(0, 1)
        return self.accuracy_per_length(pred.cpu().detach().numpy(), obs_y, obs_lens)

    def get_observation_per_digit_loss(self, model, size=OBS_SIZE):
        obs_data = self.generate_data(self.max_digits * [1 / self.max_digits], size)
        obs_X, obs_y, obs_lens = obs_data
        model.eval()
        with torch.no_grad():
            obs_X = torch.from_numpy(obs_X).float().to(model.device)
            y_pred = model(obs_X).transpose(0, 1)
            obs_loss = self.per_digit_loss(y_pred, obs_y, obs_lens, model.device)
        return obs_loss


class AdditionBanditTask(AdditionTask):
    """ This task is an addition task whose observation is prediction gain as
    in automated curriculum learning"""

    def __init__(self, *args, obs_type=OBS_TYPE, **kwargs):
        super().__init__(*args, **kwargs)
        self.set_get_observation(obs_type)

    def set_get_observation(self, obs_type):
        if obs_type == "prediction_gain":
            self.get_observation = self.get_observation_prediction_gain
        else:
            raise ValueError("OBS_TYPE is not correct")

    def get_observation_prediction_gain(self, model, last_loss, X, y, lens):
        model.eval()
        with torch.no_grad():
            pred = model(X)
            if len(pred.shape) == 2:
                pred = pred.unsqueeze(1)
            y_pred = pred.transpose(0, 1)
            current_loss = self.loss_fn(pred, y, lens)
            current_loss = current_loss.detach()

        return np.array(
            [current_loss - last_loss] * self.max_digits
        )  # all the same loss


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

