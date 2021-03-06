from collections import deque
import sys
import numpy as np
from raw_ucb.raw_ucb import EFF_RAWUCB

###############################################################################
class AbstractTeacher:
    def __init__(self, n_actions=1):
        self.n_actions = n_actions

    def give_task(self, last_rewards):
        """Returns probabilities over list of actions"""
        raise NotImplementedError


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
class ThompsonPolicy(
    EpsilonGreedyPolicy
):  # don't be alarmed, thompson policy is implemented inside sampling
    pass


class OnlineSlopeSequentialTeacher(AbstractTeacher):
    def __init__(
        self, policy=None, n_actions=1, lr=0.1, absolute=False, temperature=1.0
    ):
        super().__init__(n_actions)
        if policy is None:
            policy = BoltzmannPolicy(temperature)
        self.policy = policy
        self.lr = lr
        self.absolute = absolute
        self.Q = np.zeros(n_actions)

    def give_task(self, last_rewards):
        if last_rewards is not None:
            if np.isnan(last_rewards).any():
                np.nan_to_num(last_rewards, copy=False, nan=0.0)
            self.Q = self.Q + self.lr * (last_rewards - self.Q)
        return self.policy(np.abs(self.Q) if self.absolute else self.Q)


class OnlineBanditTeacher(AbstractTeacher):
    def __init__(
        self, policy=None, n_actions=1, lr=0.1, absolute=False, temperature=1.0
    ):
        super().__init__(n_actions)
        if policy is None:
            policy = BoltzmannPolicy(temperature)
        self.policy = policy
        self.lr = lr
        self.absolute = absolute
        self.Q = np.zeros(n_actions)
        self.a = None

    def give_task(self, last_rewards):
        if last_rewards is not None:
            if np.isnan(last_rewards).any():
                np.nan_to_num(last_rewards, copy=False, nan=0.0)
            self.Q[self.a] = self.Q[self.a] + self.lr * (  # update only last action
                last_rewards[self.a] - self.Q[self.a]
            )

        p = self.policy(np.abs(self.Q) if self.absolute else self.Q)
        self.a = np.random.choice(np.arange(len(p)), size=1, p=p)
        out = np.zeros(len(p))
        out[self.a] = 1
        return out


class SamplingTeacher(AbstractTeacher):
    def __init__(
        self, policy=ThompsonPolicy(), n_actions=1, window_size=10, absolute=False
    ):
        super().__init__(n_actions)
        self.policy = policy
        self.window_size = window_size
        self.absolute = absolute
        self.dscores = deque(maxlen=window_size)
        self.prevr = np.zeros(n_actions)
        self.n_actions = n_actions

    def give_task(self, last_rewards):
        if last_rewards is not None:
            self.dscores.append(
                last_rewards
            )  # last reward should be an array with one reward per arm
        if len(self.dscores) > 1:  # if enough values in deque
            if isinstance(
                self.policy, ThompsonPolicy
            ):  # this is totally fake, thompson policy is implemented here
                slopes = [
                    np.random.choice(drs)
                    for drs in np.array(
                        self.dscores
                    ).T  # slope of (timestep, arm) before transpose, (arm, timestep) after (randomly sample a slope for each arm)
                ]
            else:
                slopes = np.mean(self.dscores, axis=0)
        else:
            slopes = np.ones(self.n_actions)
        return self.policy(np.abs(slopes) if self.absolute else slopes)


###############################################################################
###############################################################################
# RAW UCB
###############################################################################
class RAWUCBTeacher(AbstractTeacher):
    def __init__(self, n_actions=1, sigma=1, alpha=1.4):
        super().__init__(n_actions)
        self.policy = EFF_RAWUCB(n_actions, subgaussian=sigma, alpha=alpha)

    def give_task(self, last_rewards):
        """Returns probabilities over list of actions"""
        if self.policy.t != 0:
            self.policy.getReward(self.last_choice, last_rewards[self.last_choice])
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
    rewards = np.zeros(2)
    for timestep in range(problem.T):
        task_dist = teacher.give_task(rewards)
        choice = np.argmax(task_dist)
        choices.append(choice)
        rewards = problem.step(chosen_arm=choice)
        received_rewards.append(rewards[0])

    regret = np.array(
        [0.5 if c == 0 else 0 for c in choices[:7500]]
        + [0.1 if c == 1 else 0 for c in choices[7500:]]
    )
    cum_regret = np.cumsum(regret)
    received_rewards = np.array(received_rewards)
    w = 100
    plt.plot(received_rewards, ".", alpha=0.1)
    plt.plot(
        np.concatenate(
            (np.zeros(w - 1), np.convolve(received_rewards, np.ones(w) / w, "valid"))
        ), linewidth=3
    )    
    plt.plot(
        np.concatenate((np.ones(7500), np.ones(problem.T - 7500) * 0.4)), linewidth=3
    )
    plt.plot(np.ones(len(received_rewards)) * 0.5, linewidth=3)

    plt.plot(regret, linewidth=3)
    plt.plot(cum_regret / cum_regret[-1], linewidth=3)
    plt.legend(
        [
            "Received rewards",
            "Window averaged rewards with w={}".format(w),
            "Expected reward arm 1",
            "Expected reward arm 2",
            "Regret",
            "Cummulative regret",
        ]
    )
    plt.xlabel("Iterations")
    plt.ylabel("Reward")
    plt.title("RAW-UCB Algorithm")
    plt.savefig("test_raw_ucb.png", dpi=500)


###############################################################################
class CurriculumTeacher(AbstractTeacher):
    def __init__(self, curriculum, curriculum_schedule=None, n_actions=1):
        """
        'curriculum' e.g. list of lists as defined in DIGITS_DIST_EXPERIMENTS
        """
        super().__init__(n_actions)
        self.curriculum = curriculum
        self.curriculum_step = 0
        self.interaction = 0
        self.curriculum_schedule = curriculum_schedule

    def give_task(self, last_rewards):
        self.interaction += 1
        if self.check_next_curriculum_step():
            self.curriculum_step = min(
                self.curriculum_step + 1, len(self.curriculum) - 1
            )
        p = self.curriculum[self.curriculum_step]
        return p

    def check_next_curriculum_step(self):
        """
        check whether or not increasing the curriculum step
        """
        # first check if not None or empty
        if (
            self.curriculum_schedule is not None
            and len(self.curriculum_schedule) > self.curriculum_step
            and self.interaction < self.curriculum_schedule[self.curriculum_step]
        ):
            return False
        else:
            self.interaction = 0  # reset counter
            return True


###############################################################################

"""Franco comment:
This first set of functions generate "curriculums". A curriculum is a
list of discrete probability distributions over the possible digits. For 
instance, a curriculum using a uniform pdf over the subtasks (1, 2) and then
training on the final task (3) should be [[1/2, 1/2, 0], [0, 0, 1]]. The
assertions below show the different curricula here defined. The last element
of the list is the validation distribution."""

DIGITS_DIST_EXPERIMENTS = {  # example curricula for 4 actions
    "direct": [[0, 0, 0, 1]],
    "baseline": [[1 / 4, 1 / 4, 1 / 4, 1 / 4]],
    "incremental": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
    "naive": [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [1 / 4, 1 / 4, 1 / 4, 1 / 4],
    ],
    "mixed": [
        [1, 0, 0, 0],
        [1 / 2, 1 / 2, 0, 0],
        [1 / 3, 1 / 3, 1 / 3, 0],
        [1 / 4, 1 / 4, 1 / 4, 1 / 4],
    ],
    "combined": [
        [1, 0, 0, 0],
        [1 / 4, 3 / 4, 0, 0],
        [1 / 6, 1 / 6, 2 / 3, 0],
        [1 / 8, 1 / 8, 1 / 8, 5 / 8],
        [1 / 4, 1 / 4, 1 / 4, 1 / 4],
    ],
}


def gen_curriculum_direct(gen_digits):
    return [[0 if i < gen_digits - 1 else 1 for i in range(gen_digits)]]


def gen_curriculum_baseline(gen_digits):
    return [[1 / gen_digits for _ in range(gen_digits)]]


def gen_curriculum_incremental(gen_digits):
    return [[1 if i == j else 0 for j in range(gen_digits)] for i in range(gen_digits)]


def gen_curriculum_naive(gen_digits):
    return [
        [1 if i == j else 0 for j in range(gen_digits)] for i in range(gen_digits)
    ] + gen_curriculum_baseline(gen_digits)


def gen_curriculum_mixed(gen_digits):
    return [
        [1 / (i + 1) if j <= i else 0 for j in range(gen_digits)]
        for i in range(gen_digits)
    ]


def gen_curriculum_combined(gen_digits):
    return [
        [
            1 / (2 * (i + 1)) if j < i else 1 / 2 + 1 / (2 * (i + 1)) if i == j else 0
            for j in range(gen_digits)
        ]
        for i in range(gen_digits)
    ] + gen_curriculum_baseline(gen_digits)


assert gen_curriculum_direct(4) == DIGITS_DIST_EXPERIMENTS["direct"]
assert gen_curriculum_baseline(4) == DIGITS_DIST_EXPERIMENTS["baseline"]
assert gen_curriculum_incremental(4) == DIGITS_DIST_EXPERIMENTS["incremental"]
assert gen_curriculum_naive(4) == DIGITS_DIST_EXPERIMENTS["naive"]
assert gen_curriculum_mixed(4) == DIGITS_DIST_EXPERIMENTS["mixed"]
assert gen_curriculum_combined(4) == DIGITS_DIST_EXPERIMENTS["combined"]
###############################################################################
###############################################################################
