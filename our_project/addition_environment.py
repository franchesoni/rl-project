import argparse
import csv
import os
import random

import numpy as np
import torch
from collections import deque, defaultdict
from keras.layers import TimeDistributed, Dense, RepeatVector, recurrent, Input
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
import keras.backend as K


'''Franco comment:
This first set of functions generate "curriculums". A curriculum is a
list of discrete probability distributions over the possible digits. For 
instance, a curriculum using a uniform pdf over the subtasks (1, 2) and then
training on the final task (3) should be [[1/2, 1/2, 0], [0, 0, 1]]. The
assertions below show the different curricula here defined. The last element
of the list is the validation distribution.'''

def gen_curriculum_baseline(gen_digits):
    return [[1/gen_digits for _ in range(gen_digits)]]


def gen_curriculum_naive(gen_digits):
    return [[1 if i == j else 0 for j in range(gen_digits)] for i in range(gen_digits)] + gen_curriculum_baseline(gen_digits)


def gen_curriculum_mixed(gen_digits):
    return [[1/(i+1) if j <= i else 0 for j in range(gen_digits)] for i in range(gen_digits)]


def gen_curriculum_combined(gen_digits):
    return [[1/(2*(i+1)) if j < i else 1/2 + 1/(2*(i+1)) if i == j else 0 for j in range(gen_digits)] for i in range(gen_digits)] + gen_curriculum_baseline(gen_digits)


DIGITS_DIST_EXPERIMENTS = {
    'baseline': [[1/4, 1/4, 1/4, 1/4]],
    'naive': [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [1/4, 1/4, 1/4, 1/4]],
    'mixed': [[1, 0, 0, 0], [1/2, 1/2, 0, 0], [1/3, 1/3, 1/3, 0], [1/4, 1/4, 1/4, 1/4]],
    'combined': [[1, 0, 0, 0], [1/4, 3/4, 0, 0], [1/6, 1/6, 2/3, 0], [1/8, 1/8, 1/8, 5/8], [1/4, 1/4, 1/4, 1/4]],
}
assert gen_curriculum_baseline(4) == DIGITS_DIST_EXPERIMENTS['baseline']
assert gen_curriculum_naive(4) == DIGITS_DIST_EXPERIMENTS['naive']
assert gen_curriculum_mixed(4) == DIGITS_DIST_EXPERIMENTS['mixed']
assert gen_curriculum_combined(4) == DIGITS_DIST_EXPERIMENTS['combined']


###################


class CharacterTable(object):
    '''
    Given a set of characters:
    + Encode them to a one hot integer representation
    + Decode the one hot integer representation to their character output
    + Decode a vector of probabilities to their character output
    '''
    def __init__(self, chars, maxlen):  # one hot encoded vector dim: (maxlen, #chars)
        self.chars = sorted(set(chars))  # sorted list of chars without duplicates
        self.from_char_indices = dict((c, i) for i, c in enumerate(self.chars))  # access dict with char to find index as x_ind = self.char_indices['x']
        self.from_indices_char = dict((i, c) for i, c in enumerate(self.chars))  # access dict with index to find char. It should be the same than self.chars[ind]
        self.maxlen = maxlen

    def encode(self, C, maxlen=None):  # encode string C as one-hot encoding
        assert type(C) == str  # check type
        maxlen = maxlen or self.maxlen  # overwrite maxlen if passed
        assert len(C) <= maxlen  # we can not encode a sequence larger than maxlen
        X = np.zeros((maxlen, len(self.chars)))  # one hot encoding dim: (maxlen, #chars)
        for i, c in enumerate(C):  # for each character in string C
            X[i, self.from_char_indices[c]] = 1  # set a 1 in ith row and corresponding column
        return X

    def decode(self, X):   
        X = X.argmax(axis=-1)  # gets first index of greatest integer
        return ''.join(self.from_indices_char[x] for x in X)


class AdditionRNNModel(object):
    def __init__(self, max_digits=15, hidden_size=128, batch_size=4096, invert=True, optimizer_lr=0.001, clipnorm=None, logdir=None):
        self.max_digits = max_digits
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.invert = invert
        self.optimizer_lr = optimizer_lr
        self.clipnorm = clipnorm
        self.logdir = logdir
        self.maxlen = max_digits + 1 + max_digits

        self.chars = '0123456789+ '
        self.num_chars = len(self.chars)
        self.ctable = CharacterTable(self.chars, self.maxlen)

        self.epochs = 0
        self.make_model()


    def make_model(self):
        input = Input(shape=(self.maxlen, self.num_chars))
        # "Encode" the input sequence using an RNN, producing an output of HIDDEN_SIZE
        # note: in a situation where your input sequences have a variable length,
        # use input_shape=(None, nb_feature).
        x = recurrent.LSTM(self.hidden_size)(input)
        # For the decoder's input, we repeat the encoded input for each time step
        x = RepeatVector(self.max_digits + 1)(x)
        # The decoder RNN could be multiple layers stacked or a single layer
        x = recurrent.LSTM(self.hidden_size, return_sequences=True)(x)
        # For each of step of the output sequence, decide which character should be chosen
        x = TimeDistributed(Dense(self.num_chars, activation='softmax'))(x)

        def full_number_accuracy(y_true, y_pred):
            '''Accuracy in the sense of y_true and y_pred being identical'''
            y_true_argmax = K.argmax(y_true)
            y_pred_argmax = K.argmax(y_pred)
            tfd = K.equal(y_true_argmax, y_pred_argmax)
            tfn = K.all(tfd, axis=1)
            tfc = K.cast(tfn, dtype='float32')
            tfm = K.mean(tfc)
            return tfm

        self.model = Model(inputs=input, outputs=x)
        self.model.compile(
            loss='categorical_crossentropy',
            optimizer=Adam(lr=self.optimizer_lr, clipnorm=self.clipnorm),
            metrics=['accuracy', full_number_accuracy])

    def generate_data(self, dist, size):
        '''Generates onehot encoded batch X of shape (size, maxlen, num_chars)
        and y of shape (size, max_digits+1, num_chars). The order of the one hot
        encoding is reversed: "2021+20  " is passed as the onehot encoded version
        of "  02+1202". The answer is " 2041" and is encoded without inversion.'''
        questions = []
        expected = []
        lengths = []
        while len(questions) < size:
            gen_digits = 1 + np.random.choice(len(dist), p=dist)  # get a random length in digits according to pdf "dist"
            f = lambda: int(''.join(np.random.choice(list('0123456789')) for i in range(gen_digits)))  # random number generator function
            a, b = f(), f()

            # Pad the data with spaces such that it is always MAXLEN
            q = '{}+{}'.format(a, b)
            query = q + ' ' * (self.maxlen - len(q))
            ans = str(a + b)
            # Answers can be of maximum size DIGITS + 1
            ans += ' ' * (self.max_digits + 1 - len(ans))
            if self.invert:
                query = query[::-1]  # spaces first. It could have zeros first too.
            questions.append(query)
            expected.append(ans)
            lengths.append(gen_digits)

        X = np.zeros((len(questions), self.maxlen, self.num_chars), dtype=np.bool)
        y = np.zeros((len(questions), self.max_digits + 1, self.num_chars), dtype=np.bool)
        for i, sentence in enumerate(questions):
            X[i] = self.ctable.encode(sentence, maxlen=self.maxlen)
        for i, sentence in enumerate(expected):
            y[i] = self.ctable.encode(sentence, maxlen=self.max_digits + 1)

        return X, y, np.array(lengths)

    def accuracy_per_length(self, X, y, lengths):
        '''Computes accuracy using model and the output of "generate_data".'''
        # # todo: get rid of this, should be part of train FF pass
        p = self.model.predict(X, batch_size=self.batch_size)  # output probas
        NotImplementedError('Use pytorch!!')

        y = np.argmax(y, axis=-1)  # target indices  (batch, max_digits+1) 
        p = np.argmax(p, axis=-1)  # inferred indices  (batch, max_digits+1)

        accs = []
        for i in range(self.max_digits):
            yl = y[lengths == i+1]  # select those labels with length 
            pl = p[lengths == i+1]
            tf = np.all(yl == pl, axis=1)  # set to true those that coincide
            accs.append(np.mean(tf))

        return np.array(accs)

    def train_epoch(self, train_data, val_data=None):
        train_X, train_y, train_lens = train_data
        if val_data is not None:
            val_X, val_y, val_lens = val_data

        NotImplementedError('Use pytorch!!')
        # here we should have the pytorch training loop over epochs
        history = self.model.fit(
            train_X, train_y,
            batch_size=self.batch_size,
            epochs=self.epochs + 1,
            validation_data=(val_X, val_y) if val_data else None,
            initial_epoch=self.epochs,
            callbacks=self.callbacks
        )
        self.epochs += 1
        return history.history

class AdditionLSTM(torch.nn.Module):
    def __init__(self, onehot_features=12, hidden_size=128, max_digits=15):
        '''
        onehot_features is 12 because len('0123456789+ ') = 12

        Creates encoding layer. It will take as input (batch, seq, feature) and
        will output (batch, hidden_dim) values (we will use h_n)

        Creates decoding layer. It will take as input (seq=max_digits+1, batch, hidden_dim).
        The output is h_ts of shape (max_digits+1, 1, hidden_dim).

        Finally a linear layer applied in the time dimension: it takes 
        (max_digits+1, hidden_dim) and outputs (max_digits+1, onehot_features).
        + Softmax
        '''
        super().__init__()
        self.max_digits = max_digits
        self.lstm_encode = torch.nn.LSTM(input_size=onehot_features, hidden_size=hidden_size, batch_first=True)
        self.lstm_decode = torch.nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, batch_first=False)
        self.linear = torch.nn.Linear(in_features=128, out_features=onehot_features)


    def forward(self, encoded_input):
        # breakpoint()
        h_ts, (h_n, c_n) = self.lstm_encode(encoded_input)
        expanded_h_n = h_n.expand((self.max_digits + 1, h_n.shape[1],h_n.shape[2])).contiguous()  # creates a brand new tensor
        h_ts, (h_n, c_n) = self.lstm_decode(expanded_h_n)  # we change batch order: (seq, batch, feature)
        pre_probas = self.linear(torch.squeeze(h_ts))  # squeeze in order to compute softmax in the correct axis
        return torch.nn.functional.softmax(pre_probas, dim=1)


def test_CharacterTable():
    charTable = CharacterTable('abcd ', 10)
    input_C = 'da da'
    # test encode
    # first, 'd' = 5, 'a' = 1, ' ' = 0 and fill with zeros for 5 more characters
    expected_code1 = np.array([[0, 0, 0, 0, 1], [0, 1, 0, 0, 0], [1, 0, 0, 0, 0], [0, 0, 0, 0, 1], [0, 1, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]])
    expected_code2 = np.array([[0, 0, 0, 0, 1], [0, 1, 0, 0, 0], [1, 0, 0, 0, 0], [0, 0, 0, 0, 1], [0, 1, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]])
    actual_code1 = charTable.encode(input_C)
    actual_code2 = charTable.encode(input_C, maxlen=7)
    assert np.all(expected_code1 == actual_code1)
    assert np.all(expected_code2 == actual_code2)
    # test decode
    input_code1 = expected_code1
    actual_decoded_str = charTable.decode(input_code1)
    assert actual_decoded_str[:len(input_C)] == input_C  # ignore the padded spaces
    print('Great! CharacterTable works as expected')



def test_make_model():
    chars = '0123456789+ '
    input_C = '12   +   34'
    maxlen = 11
    max_digits = 5
    num_chars = len(sorted(set(chars)))
    charTable = CharacterTable(chars, maxlen)
    encoded_input = charTable.encode(input_C)

    hidden_size = 128

    input = Input(shape=(maxlen, num_chars))
    # "Encode" the input sequence using an RNN, producing an output of HIDDEN_SIZE
    # note: in a situation where your input sequences have a variable length,
    # use input_shape=(None, nb_feature).
    x = recurrent.LSTM(hidden_size)(input)  # outputs only one hidden_size vector
    # For the decoder's input, we repeat the encoded input for each time step
    x = RepeatVector(max_digits + 1)(x)  # at most one digit more than the maximum of digits of the sumands
    # The decoder RNN could be multiple layers stacked or a single layer
    x = recurrent.LSTM(hidden_size, return_sequences=True)(x)  # outputs every hidden state
    # For each of step of the output sequence, decide which character should be chosen
    x = TimeDistributed(Dense(num_chars, activation='softmax'))(x)

    model = Model(inputs=input, outputs=x)

    # test model
    output = model(np.array([encoded_input]))
    assert output.shape == (1, max_digits+1, num_chars)
    print('Great! The keras model outputs something that makes sense')



def test_softmax():
    input_ones = torch.ones((5, 4))
    assert (input_ones / 4 == torch.nn.functional.softmax(input_ones, dim=1)).byte().all()
    assert (input_ones / 5 == torch.nn.functional.softmax(input_ones, dim=0)).byte().all()
    print('Great! Softmax works as expected.')


def test_AdditionLSTM():
    # create input
    chars = '0123456789+ '
    input_C = '12+304'
    max_digits = 4
    maxlen = max_digits * 2 + 1
    charTable = CharacterTable(chars, maxlen)
    encoded_input = charTable.encode(input_C)

    # set as tensor, initialize and forward pass
    torch_input = torch.from_numpy(np.array([encoded_input])).float()
    model = AdditionLSTM(max_digits=max_digits)
    output = model(torch_input)
    assert output.shape == (max_digits+1, len(sorted(set(chars))))





if __name__ == '__main__':
    test_CharacterTable()
    # test_make_model()
    test_softmax()
    test_AdditionLSTM()


class AdditionRNNEnvironment:
    def __init__(self, model, train_size, val_size, val_dist, writer=None):
        self.model = model
        self.num_actions = model.max_digits
        self.train_size = train_size
        self.val_data = model.generate_data(val_dist, val_size)
        self.writer = writer

    def step(self, train_dist):
        print("Training on", train_dist)
        train_data = self.model.generate_data(train_dist, self.train_size)
        history = self.model.train_epoch(train_data, self.val_data)
        #train_accs = self.model.accuracy_per_length(*train_data)
        val_accs = self.model.accuracy_per_length(*self.val_data)

        train_done = history['full_number_accuracy'][-1] > 0.99
        val_done = history['val_full_number_accuracy'][-1] > 0.99

        if self.writer:
            for k, v in history.items():
                add_summary(self.writer, "model/" + k, v[-1], self.model.epochs)
            for i in range(self.num_actions):
                #add_summary(self.writer, "train_accuracies/task_%d" % (i + 1), train_accs[i], self.model.epochs)
                add_summary(self.writer, "valid_accuracies/task_%d" % (i + 1), val_accs[i], self.model.epochs)

        return val_accs, train_done, val_done

# #################
# class EpsilonGreedyPolicy:
#     def __init__(self, epsilon=0.01):
#         self.epsilon = epsilon

#     def __call__(self, Q):
#         # find the best action with random tie-breaking
#         idx = np.where(Q == np.max(Q))[0]
#         assert len(idx) > 0, str(Q)
#         a = np.random.choice(idx)

#         # create a probability distribution
#         p = np.zeros(len(Q))
#         p[a] = 1

#         # Mix in a uniform distribution, to do exploration and
#         # ensure we can compute slopes for all tasks
#         p = p * (1 - self.epsilon) + self.epsilon / p.shape[0]

#         assert np.isclose(np.sum(p), 1)
#         return p


# class BoltzmannPolicy:
#     def __init__(self, temperature=1.0):
#         self.temperature = temperature

#     def __call__(self, Q):
#         e = np.exp((Q - np.max(Q)) / self.temperature)
#         p = e / np.sum(e)

#         assert np.isclose(np.sum(p), 1)
#         return p


# # HACK: just use the class name to detect the policy
# class ThompsonPolicy(EpsilonGreedyPolicy):
#     pass


# def estimate_slope(x, y):
#     assert len(x) == len(y)
#     A = np.vstack([x, np.ones(len(x))]).T
#     c, _ = np.linalg.lstsq(A, y)[0]
#     return c


# class CurriculumTeacher:
#     def __init__(self, env, curriculum, writer=None):
#         """
#         'curriculum' e.g. arrays defined in addition_rnn_curriculum.DIGITS_DIST_EXPERIMENTS
#         """
#         self.env = env
#         self.curriculum = curriculum
#         self.writer = writer

#     def teach(self, num_timesteps=2000):
#         curriculum_step = 0
#         for t in range(num_timesteps):
#             p = self.curriculum[curriculum_step]
#             r, train_done, val_done = self.env.step(p)
#             if (
#                 train_done
#                 and curriculum_step < len(self.curriculum) - 1
#             ):
#                 curriculum_step = curriculum_step + 1
#             if val_done:
#                 return self.env.model.epochs

#             if self.writer:
#                 for i in range(self.env.num_actions):
#                     tensorboard_utils.add_summary(
#                         self.writer,
#                         "probabilities/task_%d" % (i + 1),
#                         p[i],
#                         self.env.model.epochs,
#                     )

#         return self.env.model.epochs


# class NaiveSlopeBanditTeacher:
#     def __init__(
#         self,
#         env,
#         policy,
#         lr=0.1,
#         window_size=10,
#         abs=False,
#         writer=None,
#     ):
#         self.env = env
#         self.policy = policy
#         self.lr = lr
#         self.window_size = window_size
#         self.abs = abs
#         self.Q = np.zeros(env.num_actions)
#         self.writer = writer

#     def teach(self, num_timesteps=2000):
#         for t in range(num_timesteps // self.window_size):
#             p = self.policy(np.abs(self.Q) if self.abs else self.Q)
#             scores = [[] for _ in range(len(self.Q))]
#             for i in range(self.window_size):
#                 r, train_done, val_done = self.env.step(p)
#                 if val_done:
#                     return self.env.model.epochs
#                 for a, score in enumerate(r):
#                     if not np.isnan(score):
#                         scores[a].append(score)
#             s = [
#                 estimate_slope(list(range(len(sc))), sc)
#                 if len(sc) > 1
#                 else 1
#                 for sc in scores
#             ]
#             self.Q += self.lr * (s - self.Q)

#             if self.writer:
#                 for i in range(self.env.num_actions):
#                     tensorboard_utils.add_summary(
#                         self.writer,
#                         "Q_values/task_%d" % (i + 1),
#                         self.Q[i],
#                         self.env.model.epochs,
#                     )
#                     tensorboard_utils.add_summary(
#                         self.writer,
#                         "slopes/task_%d" % (i + 1),
#                         s[i],
#                         self.env.model.epochs,
#                     )
#                     tensorboard_utils.add_summary(
#                         self.writer,
#                         "probabilities/task_%d" % (i + 1),
#                         p[i],
#                         self.env.model.epochs,
#                     )

#         return self.env.model.epochs


# class OnlineSlopeBanditTeacher:
#     def __init__(self, env, policy, lr=0.1, abs=False, writer=None):
#         self.env = env
#         self.policy = policy
#         self.lr = lr
#         self.abs = abs
#         self.Q = np.zeros(env.num_actions)
#         self.prevr = np.zeros(env.num_actions)
#         self.writer = writer

#     def teach(self, num_timesteps=2000):
#         for t in range(num_timesteps):
#             p = self.policy(np.abs(self.Q) if self.abs else self.Q)
#             r, train_done, val_done = self.env.step(p)
#             if val_done:
#                 return self.env.model.epochs
#             s = r - self.prevr

#             # safeguard against not sampling particular action at all
#             s = np.nan_to_num(s)
#             self.Q += self.lr * (s - self.Q)
#             self.prevr = r

#             if self.writer:
#                 for i in range(self.env.num_actions):
#                     tensorboard_utils.add_summary(
#                         self.writer,
#                         "Q_values/task_%d" % (i + 1),
#                         self.Q[i],
#                         self.env.model.epochs,
#                     )
#                     tensorboard_utils.add_summary(
#                         self.writer,
#                         "slopes/task_%d" % (i + 1),
#                         s[i],
#                         self.env.model.epochs,
#                     )
#                     tensorboard_utils.add_summary(
#                         self.writer,
#                         "probabilities/task_%d" % (i + 1),
#                         p[i],
#                         self.env.model.epochs,
#                     )

#         return self.env.model.epochs


# class WindowedSlopeBanditTeacher:
#     def __init__(
#         self, env, policy, window_size=10, abs=False, writer=None
#     ):
#         self.env = env
#         self.policy = policy
#         self.window_size = window_size
#         self.abs = abs
#         self.scores = [
#             deque(maxlen=window_size) for _ in range(env.num_actions)
#         ]
#         self.timesteps = [
#             deque(maxlen=window_size) for _ in range(env.num_actions)
#         ]
#         self.writer = writer

#     def teach(self, num_timesteps=2000):
#         for t in range(num_timesteps):
#             slopes = [
#                 estimate_slope(timesteps, scores)
#                 if len(scores) > 1
#                 else 1
#                 for timesteps, scores in zip(
#                     self.timesteps, self.scores
#                 )
#             ]
#             p = self.policy(np.abs(slopes) if self.abs else slopes)
#             r, train_done, val_done = self.env.step(p)
#             if val_done:
#                 return self.env.model.epochs
#             for a, s in enumerate(r):
#                 if not np.isnan(s):
#                     self.scores[a].append(s)
#                     self.timesteps[a].append(t)

#             if self.writer:
#                 for i in range(self.env.num_actions):
#                     tensorboard_utils.add_summary(
#                         self.writer,
#                         "slopes/task_%d" % (i + 1),
#                         slopes[i],
#                         self.env.model.epochs,
#                     )
#                     tensorboard_utils.add_summary(
#                         self.writer,
#                         "probabilities/task_%d" % (i + 1),
#                         p[i],
#                         self.env.model.epochs,
#                     )

#         return self.env.model.epochs


# class SamplingTeacher:
#     def __init__(
#         self, env, policy, window_size=10, abs=False, writer=None
#     ):
#         self.env = env
#         self.policy = policy
#         self.window_size = window_size
#         self.abs = abs
#         self.writer = writer
#         self.dscores = deque(maxlen=window_size)
#         self.prevr = np.zeros(self.env.num_actions)

#     def teach(self, num_timesteps=2000):
#         for t in range(num_timesteps):
#             # find slopes for each task
#             if len(self.dscores) > 0:
#                 if isinstance(self.policy, ThompsonPolicy):
#                     slopes = [
#                         np.random.choice(drs)
#                         for drs in np.array(self.dscores).T
#                     ]
#                 else:
#                     slopes = np.mean(self.dscores, axis=0)
#             else:
#                 slopes = np.ones(self.env.num_actions)

#             p = self.policy(np.abs(slopes) if self.abs else slopes)
#             r, train_done, val_done = self.env.step(p)
#             if val_done:
#                 return self.env.model.epochs

#             # log delta score
#             dr = r - self.prevr
#             self.prevr = r
#             self.dscores.append(dr)

#             if self.writer:
#                 for i in range(self.env.num_actions):
#                     tensorboard_utils.add_summary(
#                         self.writer,
#                         "slopes/task_%d" % (i + 1),
#                         slopes[i],
#                         self.env.model.epochs,
#                     )
#                     tensorboard_utils.add_summary(
#                         self.writer,
#                         "probabilities/task_%d" % (i + 1),
#                         p[i],
#                         self.env.model.epochs,
#                     )

#         return self.env.model.epochs



# def get_args():
#     parser = argparse.ArgumentParser()

#     parser.add_argument(
#         "--teacher",
#         choices=[
#             "curriculum",
#             "naive",
#             "online",
#             "window",
#             "sampling",
#         ],
#         default="sampling",
#     )
#     parser.add_argument(
#         "--curriculum",
#         choices=["uniform", "naive", "mixed", "combined"],
#         default="combined",
#     )
#     parser.add_argument(
#         "--policy",
#         choices=["egreedy", "boltzmann", "thompson"],
#         default="thompson",
#     )
#     parser.add_argument("--epsilon", type=float, default=0.1)
#     parser.add_argument("--temperature", type=float, default=0.0004)
#     parser.add_argument("--bandit_lr", type=float, default=0.1)
#     parser.add_argument("--window_size", type=int, default=10)
#     parser.add_argument("--abs", action="store_true", default=False)
#     parser.add_argument("--no_abs", action="store_false", dest="abs")
#     parser.add_argument("--max_timesteps", type=int, default=20000)
#     parser.add_argument("--max_digits", type=int, default=9)
#     parser.add_argument(
#         "--invert", action="store_true", default=True
#     )
#     parser.add_argument(
#         "--no_invert", action="store_false", dest="invert"
#     )
#     parser.add_argument("--hidden_size", type=int, default=128)
#     parser.add_argument("--batch_size", type=int, default=4096)
#     parser.add_argument("--train_size", type=int, default=40960)
#     parser.add_argument("--val_size", type=int, default=4096)
#     parser.add_argument("--optimizer_lr", type=float, default=0.001)
#     parser.add_argument("--clipnorm", type=float, default=2)
#     parser.add_argument("--logdir", default="addition")
#     parser.add_argument("run_id")
#     parser.add_argument("--csv_file")
#     args = parser.parse_args()
#     return args

# if __name__ == "__main__":
#     args = get_args()

#     logdir = os.path.join(args.logdir, args.run_id)
#     writer = tensorboard_utils.create_summary_writer(logdir)

#     model = addition_rnn_model.AdditionRNNModel(
#         args.max_digits,
#         args.hidden_size,
#         args.batch_size,
#         args.invert,
#         args.optimizer_lr,
#         args.clipnorm,
#     )

#     val_dist = addition_rnn_curriculum.gen_curriculum_baseline(args.max_digits)[-1]
#     env = addition_rnn_model.AdditionRNNEnvironment(
#         model, args.train_size, args.val_size, val_dist, writer
#     )

#     if args.teacher != "curriculum":
#         if args.policy == "egreedy":
#             policy = EpsilonGreedyPolicy(args.epsilon)
#         elif args.policy == "boltzmann":
#             policy = BoltzmannPolicy(args.temperature)
#         elif args.policy == "thompson":
#             assert (
#                 args.teacher == "sampling"
#             ), "ThompsonPolicy can be used only with SamplingTeacher."
#             policy = ThompsonPolicy(args.epsilon)
#         else:
#             assert False

#     if args.teacher == "naive":
#         teacher = NaiveSlopeBanditTeacher(
#             env,
#             policy,
#             args.bandit_lr,
#             args.window_size,
#             args.abs,
#             writer,
#         )
#     elif args.teacher == "online":
#         teacher = OnlineSlopeBanditTeacher(
#             env, policy, args.bandit_lr, args.abs, writer
#         )
#     elif args.teacher == "window":
#         teacher = WindowedSlopeBanditTeacher(
#             env, policy, args.window_size, args.abs, writer
#         )
#     elif args.teacher == "sampling":
#         teacher = SamplingTeacher(
#             env, policy, args.window_size, args.abs, writer
#         )
#     elif args.teacher == "curriculum":
#         if args.curriculum == "uniform":
#             curriculum = addition_rnn_curriculum.gen_curriculum_baseline(args.max_digits)
#         elif args.curriculum == "naive":
#             curriculum = addition_rnn_curriculum.gen_curriculum_naive(args.max_digits)
#         elif args.curriculum == "mixed":
#             curriculum = addition_rnn_curriculum.gen_curriculum_mixed(args.max_digits)
#         elif args.curriculum == "combined":
#             curriculum = addition_rnn_curriculum.gen_curriculum_combined(args.max_digits)
#         else:
#             assert False

#         teacher = CurriculumTeacher(env, curriculum, writer)
#     else:
#         assert False

#     epochs = teacher.teach(args.max_timesteps)
#     print("Finished after", epochs, "epochs.")

#     if args.csv_file:
#         data = vars(args)
#         data["epochs"] = epochs
#         header = sorted(data.keys())

#         # write the CSV file one directory above the experiment directory
#         csv_file = os.path.join(args.logdir, args.csv_file)
#         file_exists = os.path.isfile(csv_file)
#         with open(csv_file, "a") as file:
#             writer = csv.DictWriter(
#                 file, delimiter=",", fieldnames=header
#             )
#             if not file_exists:
#                 writer.writeheader()
#             writer.writerow(data)

# ################ 
# ## curriculum main
# # if __name__ == '__main__':
# #     parser = argparse.ArgumentParser()

# #     parser.add_argument('run_id')
# #     parser.add_argument('--curriculum', choices=['baseline', 'naive', 'mixed', 'combined'], default='baseline')
# #     parser.add_argument('--max_timesteps', type=int, default=2000)
# #     parser.add_argument('--max_digits', type=int, default=4)
# #     parser.add_argument('--invert', action='store_true', default=True)
# #     parser.add_argument('--no_invert', action='store_false', dest='invert')
# #     parser.add_argument('--hidden_size', type=int, default=128)
# #     parser.add_argument('--batch_size', type=int, default=4096)
# #     parser.add_argument('--train_size', type=int, default=40960)
# #     parser.add_argument('--val_size', type=int, default=4096)
# #     parser.add_argument('--optimizer_lr', type=float, default=0.001)
# #     parser.add_argument('--clipnorm', type=float, default=2)
# #     parser.add_argument('--logdir', default='logs')
# #     args = parser.parse_args()

# #     if args.curriculum == 'baseline':
# #         curriculum_steps = gen_curriculum_baseline(args.max_digits)
# #     elif args.curriculum == 'naive':
# #         curriculum_steps = gen_curriculum_naive(args.max_digits)
# #     elif args.curriculum == 'mixed':
# #         curriculum_steps = gen_curriculum_mixed(args.max_digits)
# #     elif args.curriculum == 'combined':
# #         curriculum_steps = gen_curriculum_combined(args.max_digits)
# #     else:
# #         assert False

# #     logdir = os.path.join(args.logdir, "{0}digits-curriculum_{1}-{2}".format(args.max_digits, args.curriculum, args.run_id))
# #     writer = create_summary_writer(logdir)

# #     model = AdditionRNNModel(args.max_digits, args.hidden_size, args.batch_size, args.invert, args.optimizer_lr, args.clipnorm)

# #     val_dist = curriculum_steps[-1]
# #     env = AdditionRNNEnvironment(model, args.train_size, args.val_size, val_dist, writer)

# #     for train_dist in curriculum_steps:
# #         while model.epochs < args.max_timesteps:
# #             r, train_done, val_done = env.step(train_dist)
# #             if train_done:
# #                 break

# #     print("Finished after", model.epochs, "epochs.")
# #     assert val_done
