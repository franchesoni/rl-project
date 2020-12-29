import numpy as np
import os
import random
import torch

from cfg import BATCH_SIZE, EPOCHS, MAX_DIGITS, NUM_CHARS, TRAIN_SIZE, VAL_SIZE, WRITER
###############


'''This is the longest and ugliest code. Have a look at <AbstractClassroom>
to get an idea of the utility of having a classroom: it basically generates the
tasks signaled by the teacher and passes them to the student, then observes the
student and comes back to the teacher with a reward signal.'''


class AbstractClassroom:
    def __init__(self, teacher, student):
        self.teacher = teacher
        self.student = student
        self.reward = None

    def generate_task(self, task_index):
        """Generates a new task according to index"""
        NotImplementedError(
            "This is an abstract class, you should implement this method"
        )

    def compute_reward(self, task_index):
        """Generates a new task according to index"""
        NotImplementedError(
            "This is an abstract class, you should implement this method"
        )

    def step(self):
        task_dist = self.teacher.give_task(self.reward)
        task = self.generate_task(task_dist)
        obs = self.student.learn_from_task(task)
        self.reward = self.compute_reward(obs)

        WRITER.add_scalar('Classroom reward', self.reward)

'''The abstract task has the training parameters for the student learning. It
uses probability distributions over the tasks. This allows the teacher to give
only one task (by setting that task as with probability 1) or mix tasks. The
mixing is inside the generating data. We should see how much efficiency is lost
by reducing batch size, as it's needed for the bandit framework. Another option
is to set tasks as being a set of handcrafted curricula.'''

class AbstractTask:
    def __init__(
        self,
        train_dist,
        train_size,
        val_dist,
        val_size,
        batch_size=4096,
        epochs=1,
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
        NotImplementedError(
            "This is an abstract class, you should implement this method"
        )

    def loss_fn(self, pred, y):
        NotImplementedError(
            "This is an abstract class, you should implement this method"
        )

    def get_observation(self, model):
        """Computes the observation given model. This should be reimplemented
        inside Student if it's too inefficient to make a whole new computation."""
        NotImplementedError(
            "This is an abstract class, you should implement this method"
        )

    def val_score_fn(self, val_pred, val_y, val_lens):
        NotImplementedError(
            "This is an abstract class, you should implement this method"
        )

    def finished(self, val_score):
        """Returns bool telling if val score is enough to finish. It assumes
        the finishing criterium is based on val score. It also finishes when
        n_epochs is completed"""
        NotImplementedError(
            "This is an abstract class, you should implement this method"
        )


########################################################################################
########################################################################################
########################################################################################

# ADDITION

###############################################
class TestAdditionClassroom(AbstractClassroom):
    def __init__(self, teacher, student):
        super().__init__(teacher, student)
        self.past_obs = [0]  # initialize with something so we can take diff

    def generate_task(self, task_dist):
        """Generates a new task according to task_dist. Task dist should be
        a one hot vector when all work is finished."""
        val_dist = np.zeros_like(task_dist)
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

    def compute_reward(self, obs):
        self.past_obs.append(obs)
        return self.past_obs[-1] - self.past_obs[-2]




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
            train_dist,
            train_size,
            val_dist,
            val_size,
            batch_size,
            epochs,
        )
        self.val_dist = val_dist
        self.max_digits = max_digits
        self.invert = invert
        self.maxlen = max_digits + 1 + max_digits
        self.chars = "0123456789+ "
        self.num_chars = len(sorted(set(self.chars)))
        assert self.num_chars == NUM_CHARS
        self.ctable = CharacterTable(self.chars, self.maxlen)

    def generate_data(self, dist, size):
        """Generates onehot encoded batch X of shape (size, maxlen, num_chars)
        and y of shape (size, max_digits+1, num_chars). The order of the one hot
        encoding is reversed: "2021+20  " is passed as the onehot encoded version
        of "  02+1202". The answer is " 2041" and is encoded without inversion."""
        questions = []
        expected = []
        lengths = []
        while len(questions) < size:
            gen_digits = 1 + np.random.choice(
                len(dist), p=dist
            )  # get a random length in digits according to pdf "dist"
            f = lambda: "".join(  # this was originally wrapped inside int() but that created sums with different n of digits when randomly chosing '09' and '58' for example.
                    np.random.choice(list("0123456789"))
                    for i in range(gen_digits)
                ) # random number generator function
            a, b = f(), f()

            # Pad the data with spaces such that it is always MAXLEN
            q = "{}+{}".format(a, b)
            query = q + " " * (self.maxlen - len(q))
            ans = str(int(a) + int(b))
            # Answers can be of maximum size DIGITS + 1
            ans += " " * (self.max_digits + 1 - len(ans))
            if self.invert:
                query = query[
                    ::-1
                ]  # spaces first. It could have zeros first too.
            questions.append(query)
            expected.append(ans)
            lengths.append(gen_digits)

        X = np.zeros(
            (len(questions), self.maxlen, self.num_chars),
            dtype=np.bool,
        )
        y = np.zeros(
            (len(questions), self.max_digits + 1, self.num_chars),
            dtype=np.bool,
        )
        for i, sentence in enumerate(questions):
            X[i] = self.ctable.encode(sentence, maxlen=self.maxlen)
        for i, sentence in enumerate(expected):
            y[i] = self.ctable.encode(
                sentence, maxlen=self.max_digits + 1
            )

        WRITER.add_text('Progress', 'Generated data')

        return X, y, np.array(lengths)

    def accuracy_per_length(self, y_pred, y, lengths):
        """Computes accuracy using model output """
        y = np.argmax(
            y, axis=-1
        )  # target indices  (batch, max_digits+1)
        p = np.argmax(
            y_pred, axis=-1
        )  # inferred indices  (batch, max_digits+1)
        accs = []
        for i in range(self.max_digits):
            yl = y[
                lengths == i + 1
            ]  # select those labels with length
            pl = p[lengths == i + 1]
            tf = np.all(
                yl == pl, axis=1
            )  # set to true those that coincide
            accs.append(np.mean(tf))
        RuntimeWarning(
            "accuracy per length computed without pytorch (using numpy only)"
        )
        return np.array(accs)

    def accuracy_per_length_torch(self, y_pred, y, lengths):
        """Computes accuracy using model output """
        y = torch.argmax(
            y, dim=-1
        )  # target indices  (batch, max_digits+1)
        p = torch.argmax(
            y_pred, dim=-1
        )  # inferred indices  (batch, max_digits+1)
        accs = torch.Tensor(self.max_digits)
        for i in range(self.max_digits):
            yl = y[
                lengths == i + 1
            ]  # select those labels with length
            pl = p[lengths == i + 1]
            tf = (
                torch.all(yl == pl, dim=1)
            )  # set to true those that coincide
            accs[i] = torch.mean((tf*1).float())
        return accs

    def full_number_accuracy(self, y_pred, y_true):
        """Accuracy in the sense of y_true and y_pred being identical"""
        y_true_argmax = torch.argmax(y_true, dim=-1)
        y_pred_argmax = torch.argmax(y_pred, dim=-1)
        tfd = torch.eq(y_true_argmax, y_pred_argmax)
        tfn = torch.all(tfd, dim=1)
        tfc = tfn.float()
        tfm = torch.mean(tfc)
        return tfm

    def categorical_crossentropy(self, y_pred, y_true):
        # dimensions as in https://stackoverflow.com/questions/60121107/pytorch-nllloss-function-target-shape-mismatch
        y_true_argmax = torch.argmax(y_true, dim=-1).view(-1)  # flatten
        return torch.nn.NLLLoss()(torch.log(y_pred).view(-1, y_pred.shape[2]), y_true_argmax)

    def loss_fn(self, pred, y, lengths):
        return self.categorical_crossentropy(pred, y)

    def get_observation(self, model):
        """Computes the observation given model. This should be reimplemented
        inside Student if it's too inefficient to make a whole new computation."""
        val_data = self.generate_data(self.val_dist, 100)
        val_X, val_y, val_lens = val_data
        val_X = torch.from_numpy(val_X).float()
        val_y = torch.from_numpy(val_y).float()
        pred = model(val_X).transpose(0, 1)
        return self.val_score_fn(pred, val_y, val_lens)

    def val_score_fn(self, val_pred, val_y, val_lens):
        return self.full_number_accuracy(val_pred, val_y)

    def finished(self, val_score):
        """Returns bool telling if val score is enough to finish. It assumes
        the finishing criterium is based on val score. It also finishes when
        n_epochs is completed"""
        return False


    # use categorical crossentropy as loss


class CharacterTable(object):
    """
    Given a set of characters:
    + Encode them to a one hot integer representation
    + Decode the one hot integer representation to their character output
    + Decode a vector of probabilities to their character output
    """

    def __init__(
        self, chars, maxlen
    ):  # one hot encoded vector dim: (maxlen, #chars)
        self.chars = sorted(
            set(chars)
        )  # sorted list of chars without duplicates
        self.from_char_indices = dict(
            (c, i) for i, c in enumerate(self.chars)
        )  # access dict with char to find index as x_ind = self.char_indices['x']
        self.from_indices_char = dict(
            (i, c) for i, c in enumerate(self.chars)
        )  # access dict with index to find char. It should be the same than self.chars[ind]
        self.maxlen = maxlen

    def encode(
        self, C, maxlen=None
    ):  # encode string C as one-hot encoding
        assert type(C) == str  # check type
        maxlen = maxlen or self.maxlen  # overwrite maxlen if passed
        assert (
            len(C) <= maxlen
        )  # we can not encode a sequence larger than maxlen
        X = np.zeros(
            (maxlen, len(self.chars))
        )  # one hot encoding dim: (maxlen, #chars)
        for i, c in enumerate(C):  # for each character in string C
            X[
                i, self.from_char_indices[c]
            ] = 1  # set a 1 in ith row and corresponding column
        return X

    def decode(self, X):
        X = X.argmax(axis=-1)  # gets first index of greatest integer
        return "".join(self.from_indices_char[x] for x in X)

