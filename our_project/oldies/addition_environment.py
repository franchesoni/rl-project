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


###################




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



############################
############################
############################




