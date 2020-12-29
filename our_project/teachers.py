import numpy as np

from classroom import WRITER

'''Franco comment:
This first set of functions generate "curriculums". A curriculum is a
list of discrete probability distributions over the possible digits. For 
instance, a curriculum using a uniform pdf over the subtasks (1, 2) and then
training on the final task (3) should be [[1/2, 1/2, 0], [0, 0, 1]]. The
assertions below show the different curricula here defined. The last element
of the list is the validation distribution.'''

DIGITS_DIST_EXPERIMENTS = {  # example curricula for 4 actions
    'baseline': [[1/4, 1/4, 1/4, 1/4]],
    'naive': [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [1/4, 1/4, 1/4, 1/4]],
    'mixed': [[1, 0, 0, 0], [1/2, 1/2, 0, 0], [1/3, 1/3, 1/3, 0], [1/4, 1/4, 1/4, 1/4]],
    'combined': [[1, 0, 0, 0], [1/4, 3/4, 0, 0], [1/6, 1/6, 2/3, 0], [1/8, 1/8, 1/8, 5/8], [1/4, 1/4, 1/4, 1/4]],
}

def gen_curriculum_baseline(gen_digits):
    return [[1/gen_digits for _ in range(gen_digits)]]

def gen_curriculum_naive(gen_digits):
    return [[1 if i == j else 0 for j in range(gen_digits)] for i in range(gen_digits)] + gen_curriculum_baseline(gen_digits)

def gen_curriculum_mixed(gen_digits):
    return [[1/(i+1) if j <= i else 0 for j in range(gen_digits)] for i in range(gen_digits)]

def gen_curriculum_combined(gen_digits):
    return [[1/(2*(i+1)) if j < i else 1/2 + 1/(2*(i+1)) if i == j else 0 for j in range(gen_digits)] for i in range(gen_digits)] + gen_curriculum_baseline(gen_digits)



assert gen_curriculum_baseline(4) == DIGITS_DIST_EXPERIMENTS['baseline']
assert gen_curriculum_naive(4) == DIGITS_DIST_EXPERIMENTS['naive']
assert gen_curriculum_mixed(4) == DIGITS_DIST_EXPERIMENTS['mixed']
assert gen_curriculum_combined(4) == DIGITS_DIST_EXPERIMENTS['combined']

###############################################################################
###############################################################################
###############################################################################

class EpsilonGreedyPolicy:
    def __init__(self, epsilon=0.01):
        self.epsilon = epsilon

    def __call__(self, Q):
        # find the best action with random tie-breaking
        idx = np.where(Q == np.max(Q))[0]
        assert len(idx) > 0, str(Q)
        a = np.random.choice(idx)

        # create a probability distribution
        p = np.zeros(len(Q))
        p[a] = 1

        # Mix in a uniform distribution, to do exploration and
        # ensure we can compute slopes for all tasks
        p = p * (1 - self.epsilon) + self.epsilon / p.shape[0]

        assert np.isclose(np.sum(p), 1)
        return p


class BoltzmannPolicy:
    def __init__(self, temperature=1.0):
        self.temperature = temperature

    def __call__(self, Q):
        e = np.exp((Q - np.max(Q)) / self.temperature)
        p = e / np.sum(e)

        assert np.isclose(np.sum(p), 1)
        return p


# HACK: just use the class name to detect the policy
class ThompsonPolicy(EpsilonGreedyPolicy):
    pass


class AbstractTeacher:
    def __init__(self, n_actions=1):
        self.n_actions = n_actions

    def give_task(self, last_reward):
        '''Returns probabilities over list of actions'''
        raise NotImplementedError


import sys
sys.path.append('/home/franchesoni/Documents/mva/mva/rl/project/raw_ucb/')
from raw_ucb import EFF_RAWUCB

class RAWUCBTeacher(AbstractTeacher):
    def __init__(self, n_actions=1, sigma=1):
        super().__init__(n_actions)
        self.policy = EFF_RAWUCB(n_actions, subgaussian=sigma, alpha=1.4)

    def give_task(self, last_reward):
        '''Returns probabilities over list of actions'''
        if self.policy.t != 0:
            self.policy.getReward(self.last_choice, last_reward)
        else:
            self.policy.t += 1
        self.last_choice = self.policy.choice()
        p = np.zeros(self.n_actions)
        p[self.last_choice] = 1
        return p


def test_RAWUCB():
    import numpy as np
    import matplotlib.pyplot as plt
    from classroom import ToyRottingProblem

    received_rewards = []
    choices = []
    problem = ToyRottingProblem()
    teacher = RAWUCBTeacher(n_actions=2)
    reward = 0
    for timestep in range(problem.T):
        task_dist = teacher.give_task(reward)
        choice = np.argmax(task_dist)
        choices.append(choice)
        reward = problem.step(chosen_arm=choice)
        received_rewards.append(reward[0])

    regret = np.array([0.5 if c == 0 else 0 for c in choices[:7500]] + [0.1 if c == 1 else 0 for c in choices[7500:]])
    cum_regret = np.cumsum(regret)
    received_rewards = np.array(received_rewards)
    w = 100
    plt.plot(received_rewards, '.')
    plt.plot(np.concatenate((np.ones(7500), np.ones(problem.T-7500)*0.4)), linewidth=4)
    plt.plot(np.ones(len(received_rewards))*0.5, linewidth=4)
    plt.plot(np.concatenate((np.zeros(w-1), np.convolve(received_rewards, np.ones(w) / w, 'valid'))))
    plt.plot(regret, 'x')
    plt.plot(cum_regret / cum_regret[-1], linewidth=4)
    plt.legend(['Received rewards', 'Expected reward arm 1', 'Expected reward arm 2', f'Window averaged rewards with w={w}', 'Regret', 'Cummulative regret'])
    plt.show()






class CurriculumTeacher(AbstractTeacher):
    def __init__(self, curriculum, n_actions=1):
        """
        'curriculum' e.g. list of lists as defined in DIGITS_DIST_EXPERIMENTS
        """
        super().__init__(n_actions)
        self.curriculum = curriculum
        self.curriculum_step = 0

    def give_task(self, last_reward):
        p = self.curriculum[self.curriculum_step]
        if last_reward == 'A':  # advance when reward is 'A'
            self.curriculum_step += 1
        return p




class OnlineSlopeBanditTeacher:
    def __init__(self, policy, num_actions, lr=0.1, abs_=False):
        raise NotImplementedError
        self.policy = policy
        self.lr = lr
        self.abs_ = abs_
        self.Q = np.zeros(num_actions)
        self.prevr = np.zeros(num_actions)


    def teach(self, num_timesteps=2000):
        for t in range(num_timesteps):
            p = self.policy(np.abs(self.Q) if self.abs_ else self.Q)
            r, train_done, val_done = self.env.step(p)
            if val_done:
                return self.env.model.epochs
            s = r - self.prevr

            # safeguard against not sampling particular action at all
            s = np.nan_to_num(s)
            self.Q += self.lr * (s - self.Q)
            self.prevr = r


class SamplingTeacher:
    def __init__(
        self, env, policy, window_size=10, abs_=False, writer=None
    ):
        raise NotImplementedError
        self.env = env
        self.policy = policy
        self.window_size = window_size
        self.abs_ = abs_
        self.writer = writer
        self.dscores = deque(maxlen=window_size)
        self.prevr = np.zeros(self.env.num_actions)

    def teach(self, num_timesteps=2000):
        for t in range(num_timesteps):
            # find slopes for each task
            if len(self.dscores) > 0:
                if isinstance(self.policy, ThompsonPolicy):
                    slopes = [
                        np.random.choice(drs)
                        for drs in np.array(self.dscores).T
                    ]
                else:
                    slopes = np.mean(self.dscores, axis=0)
            else:
                slopes = np.ones(self.env.num_actions)

            p = self.policy(np.abs(slopes) if self.abs_ else slopes)
            r, train_done, val_done = self.env.step(p)
            if val_done:
                return self.env.model.epochs

            # log delta score
            dr = r - self.prevr
            self.prevr = r
            self.dscores.append(dr)

            if self.writer:
                for i in range(self.env.num_actions):
                    tensorboard_utils.add_summary(
                        self.writer,
                        "slopes/task_%d" % (i + 1),
                        slopes[i],
                        self.env.model.epochs,
                    )
                    tensorboard_utils.add_summary(
                        self.writer,
                        "probabilities/task_%d" % (i + 1),
                        p[i],
                        self.env.model.epochs,
                    )

        return self.env.model.epochs


