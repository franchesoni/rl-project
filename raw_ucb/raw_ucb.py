# -*- coding: utf-8 -*-
""" Base class for any policy.
- If rewards are not in [0, 1], be sure to give the lower value and the amplitude. Eg, if rewards are in [-3, 3], lower = -3, amplitude = 6.
"""
from __future__ import division, print_function  # Python 2 compatibility

__author__ = "Lilian Besson"
__version__ = "0.9"

import numpy as np

#: If True, every time a reward is received, a warning message is displayed if it lies outsides of ``[lower, lower + amplitude]``.
CHECKBOUNDS = True
CHECKBOUNDS = False


class BasePolicy(object):
    """ Base class for any policy."""

    def __init__(self, nbArms, lower=0., amplitude=1.):
        """ New policy."""
        # Parameters
        assert nbArms > 0, "Error: the 'nbArms' parameter of a {} object cannot be <= 0.".format(self)  # DEBUG
        self.nbArms = nbArms  #: Number of arms
        self.lower = lower  #: Lower values for rewards
        assert amplitude > 0, "Error: the 'amplitude' parameter of a {} object cannot be <= 0.".format(self)  # DEBUG
        self.amplitude = amplitude  #: Larger values for rewards
        # Internal memory
        self.t = 0  #: Internal time
        self.pulls = np.zeros(nbArms, dtype=int)  #: Number of pulls of each arms
        self.rewards = np.zeros(nbArms)  #: Cumulated rewards of each arms

    def __str__(self):
        """ -> str"""
        return self.__class__.__name__

    # --- Start game, and receive rewards

    def startGame(self):
        """ Start the game (fill pulls and rewards with 0)."""
        self.t = 0
        self.pulls.fill(0)
        self.rewards.fill(0)

    if CHECKBOUNDS:
        # XXX useless checkBounds feature
        def getReward(self, arm, reward):
            """ Give a reward: increase t, pulls, and update cumulated sum of rewards for that arm (normalized in [0, 1])."""
            self.t += 1
            self.pulls[arm] += 1
            # XXX we could check here if the reward is outside the bounds
            if not 0 <= reward - self.lower <= self.amplitude:
                print("Warning: {} received on arm {} a reward = {:.3g} that is outside the interval [{:.3g}, {:.3g}] : the policy will probably fail to work correctly...".format(self, arm, reward, self.lower, self.lower + self.amplitude))  # DEBUG
            # else:
            #     print("Info: {} received on arm {} a reward = {:.3g} that is inside the interval [{:.3g}, {:.3g}]".format(self, arm, reward, self.lower, self.lower + self.amplitude))  # DEBUG
            reward = (reward - self.lower) / self.amplitude
            self.rewards[arm] += reward
    else:
        # It's faster to define two methods and pick one
        # (one test in init, that's it)
        # rather than doing the test in the method
        def getReward(self, arm, reward):
            """ Give a reward: increase t, pulls, and update cumulated sum of rewards for that arm (normalized in [0, 1])."""
            self.t += 1
            self.pulls[arm] += 1
            reward = (reward - self.lower) / self.amplitude
            self.rewards[arm] += reward

    # --- Basic choice() and handleCollision() method

    def choice(self):
        """ Not defined."""
        raise NotImplementedError("This method choice() has to be implemented in the child class inheriting from BasePolicy.")

    # def handleCollision(self, arm, reward=None):
    #     """ Default to give a 0 reward (or ``self.lower``)."""
    #     # print("DEBUG BasePolicy.handleCollision({}, {}) was called...".format(arm, reward))  # DEBUG
    #     # self.getReward(arm, self.lower if reward is None else reward)
    #     self.getReward(arm, self.lower)
    #     # raise NotImplementedError("This method handleCollision() has to be implemented in the child class inheriting from BasePolicy.")

    # --- Others choice...() methods, partly implemented

    def choiceWithRank(self, rank=1):
        """ Not defined."""
        if rank == 1:
            return self.choice()
        else:
            raise NotImplementedError("This method choiceWithRank(rank) has to be implemented in the child class inheriting from BasePolicy.")

    def choiceFromSubSet(self, availableArms='all'):
        """ Not defined."""
        if availableArms == 'all':
            return self.choice()
        else:
            raise NotImplementedError("This method choiceFromSubSet(availableArms) has to be implemented in the child class inheriting from BasePolicy.")

    def choiceMultiple(self, nb=1):
        """ Not defined."""
        if nb == 1:
            return np.array([self.choice()])
        else:
            raise NotImplementedError("This method choiceMultiple(nb) has to be implemented in the child class inheriting from BasePolicy.")

    def choiceIMP(self, nb=1, startWithChoiceMultiple=True):
        """ Not defined."""
        if nb == 1:
            return np.array([self.choice()])
        else:
            return self.choiceMultiple(nb=nb)

    def estimatedOrder(self):
        """ Return the estimate order of the arms, as a permutation on [0..K-1] that would order the arms by increasing means.
        - For a base policy, it is completely random.
        """
        return np.random.permutation(self.nbArms)
# -*- coding: utf-8 -*-
"""
author: Julien Seznec

Filtering on Expanding Window Algorithm for rotting bandits.

Reference: [Seznec et al.,  2019a]
Rotting bandits are not harder than stochastic ones;
Julien Seznec, Andrea Locatelli, Alexandra Carpentier, Alessandro Lazaric, Michal Valko ;
Proceedings of Machine Learning Research, PMLR 89:2564-2572, 2019.
http://proceedings.mlr.press/v89/seznec19a.html
https://arxiv.org/abs/1811.11043 (updated version)

Reference : [Seznec et al.,  2019b]
A single algorithm for both rested and restless rotting bandits (WIP)
Julien Seznec, Pierre Ménard, Alessandro Lazaric, Michal Valko
"""

__author__ = "Julien Seznec"
__version__ = "0.1"

import numpy as np
import time
np.seterr(divide='ignore')  # XXX dangerous in general, controlled here!


class EFF_FEWA(BasePolicy):
  """
  Efficient Filtering on Expanding Window Average
  Efficient trick described in [Seznec et al.,  2019a, https://arxiv.org/abs/1811.11043] (m=2)
  and [Seznec et al.,  2019b, WIP] (m<=2)
  We use the confidence level :math:`\delta_t = \frac{1}{t^\alpha}`.
   """

  def __init__(self, nbArms, alpha=0.06, subgaussian=1, m=None, delta=None, delay = False):
    super(EFF_FEWA, self).__init__(nbArms)
    self.alpha = alpha
    self.nbArms = nbArms
    self.subgaussian = subgaussian
    self.delta = delta
    self.inlogconst = 1 / delta ** (1 / alpha) if delta is not None else 1
    self.armSet = np.arange(nbArms)
    self.display_m = m is not None
    self.grid = m if m is not None else 2
    self.statistics = np.ones(shape=(3, self.nbArms, 2)) * np.nan
    # [0,:,:] : current statistics, [1,:,:]: pending statistics, [2,:,:]: number of sample in the pending statistics
    self.windows = np.array([1, int(np.ceil(self.grid))])
    self.outlogconst = self._append_thresholds(self.windows)
    self.delay = np.array([0, np.nan]) if delay else []
    self.idx_nan = np.ones(nbArms)

  def __str__(self):
    if self.delta != None:
      if self.display_m:
        return r"EFF_FEWA($\alpha={:.3g}, \, \delta={:.3g}, \, m={:.3g}$)".format(self.alpha, self.delta, self.grid)
      else:
        return r"EFF_FEWA($\alpha={:.3g}, \, \delta={:.3g}$)".format(self.alpha, self.delta)
    else:
      if self.display_m:
        return r"EFF_FEWA($\alpha={:.3g}, \, m={:.3g}$)".format(self.alpha, self.grid)
      else:
        return r"EFF_FEWA($\alpha={:.3g}$)".format(self.alpha)

  def getReward(self, arm, reward):
    super(EFF_FEWA, self).getReward(arm, reward)
    if not np.all(np.isnan(self.statistics[2, :, -1])):
      add_size = (1 if self.grid > 1.005 else 3) * self.statistics.shape[2]
      self.statistics = np.append(self.statistics, np.nan * np.ones([3, self.nbArms, add_size]), axis=2)
      self.windows = np.append(self.windows, np.array(self._compute_windows(self.windows[-1], add_size), dtype=np.double))
      self.outlogconst = self._append_thresholds(self.windows)
      if len(self.delay):
        self.delay = np.append(self.delay, np.nan * np.ones(add_size))
    self.statistics[1, arm, 0] = reward
    self.statistics[2, arm, 0] = 1
    self.statistics[1, arm, 1:] += reward
    self.statistics[2, arm, 1:] += 1
    idx = np.where((self.statistics[2, arm, :] == self.windows))[0]
    if len(self.delay):
      self.delay += 1
      self.delay[idx] = 0
    self.statistics[0, arm, idx] = self.statistics[1, arm, idx]
    if int(self.idx_nan[arm] -1) in idx:
      idx = np.append(idx, int(self.idx_nan[arm]))
      self.idx_nan[arm] += 1
    self.statistics[1:, arm, idx[idx != 0]] = self.statistics[1:, arm, idx[idx != 0] - 1]

  def choice(self):
    remainingArms = self.armSet.copy()
    i = 0
    selected = remainingArms[np.isnan(self.statistics[0, :, i])] if len(
        remainingArms) != 1 else remainingArms
    sqrtlogt = np.sqrt(np.log(self._inlog()))
    while len(selected) == 0:
      thresh = np.max(self.statistics[0, remainingArms, i]) - sqrtlogt * self.outlogconst[i]
      remainingArms = remainingArms[self.statistics[0, remainingArms, i] >= thresh]
      i += 1
      selected = remainingArms[np.isnan(self.statistics[0, remainingArms, i])] if len(
        remainingArms) != 1 else remainingArms
    return selected[np.argmin(self.pulls[selected])]

  def _append_thresholds(self, w):
    return np.sqrt(8 * w * self.alpha * self.subgaussian ** 2)

  def _compute_windows(self, first_window, add_size):
      last_window = first_window
      res = []
      for i in range(add_size):
        new_window = int(np.ceil(last_window * self.grid))
        res.append(new_window)
        last_window = new_window
      return res


  def _inlog(self):
    return max(self.inlogconst * self.t, 1)

  def startGame(self):
    super(EFF_FEWA, self).startGame()
    self.statistics = np.ones(shape=(3, self.nbArms, 2)) * np.nan
    self.windows = np.array([1, int(np.ceil(self.grid))])
    self.outlogconst = self._append_thresholds(self.windows)


class FEWA(EFF_FEWA):
  """ Filtering on Expanding Window Average.
  Reference: [Seznec et al.,  2019a, https://arxiv.org/abs/1811.11043].
  FEWA is equivalent to EFF_FEWA for :math:`m < 1+1/T` [Seznec et al.,  2019b, WIP].
  This implementation is valid for $:math:`T < 10^{15}`.
  For :math:`T>10^{15}`, FEWA will have time and memory issues as its time and space complexity is O(KT) per round.
  """

  def __init__(self, nbArms, subgaussian=1, alpha=4, delta=None):
    super(FEWA, self).__init__(nbArms, subgaussian=subgaussian, alpha=alpha, delta=delta, m=1 + 10 ** (-15))

  def __str__(self):
    if self.delta != None:
      return r"FEWA($\alpha={:.3g}, \, \delta ={:.3g}$)".format(self.alpha, self.delta)
    else:
      return r"FEWA($\alpha={:.3g}$)".format(self.alpha)





"""
author: Julien Seznec

Rotting Adaptive Window Upper Confidence Bounds for rotting bandits.

Reference : [Seznec et al.,  2019b]
A single algorithm for both rested and restless rotting bandits (WIP)
Julien Seznec, Pierre Ménard, Alessandro Lazaric, Michal Valko
"""



__author__ = "Julien Seznec"
__version__ = "0.1"

import numpy as np
import time
np.seterr(divide='ignore')  # XXX dangerous in general, controlled here!



class EFF_RAWUCB(EFF_FEWA):
    """
    Efficient Rotting Adaptive Window Upper Confidence Bound (RAW-UCB) [Seznec et al.,  2020]
    Efficient trick described in [Seznec et al.,  2019a, https://arxiv.org/abs/1811.11043] (m=2)
    and [Seznec et al.,  2020] (m<=2)
    We use the confidence level :math:`\delta_t = \frac{1}{t^\alpha}`.
    """

    def choice(self):
        not_selected = np.where(self.pulls == 0)[0]
        if len(not_selected):
            return not_selected[0]
        self.ucb = self._compute_ucb()
        return np.nanmin(self.ucb, axis=1).argmax()

    def _compute_ucb(self):
        return (self.statistics[0, :, :] / self.windows + self.outlogconst * np.sqrt(np.log(self._inlog())))

    def _append_thresholds(self, w):
        # FEWA use two confidence bounds. Hence, the outlogconst is twice smaller for RAWUCB
        return np.sqrt(2 * self.alpha * self.subgaussian ** 2 / w)

    def __str__(self):
        return r"EFF_RAW-UCB($\alpha={:.3g}, \, m={:.3g}$)".format(self.alpha, self.grid)




    

if __name__ == "__main__":
    # Code for debugging purposes.
    start = time.time()
    HORIZON = 10**4
    sigma = 1
    reward = {0: 0, 1: 0.2, 2: 0.4, 3: 0.6, 4: 0.8}
    policy = EFF_RAWUCB(5, subgaussian=sigma, alpha=1.4)
    for t in range(HORIZON):
        choice = policy.choice()
        policy.getReward(choice, reward[choice])
    print(time.time() - start)
    print(policy.windows[:10])
    print(policy.outlogconst[:10])
    print(policy.pulls)