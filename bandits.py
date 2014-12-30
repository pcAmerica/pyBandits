"""
This module implements Multi-Armed Bandit algorithms as described in
John Myles White's book, Bandit Algorithms for Website Optimization.

see: https://github.com/johnmyleswhite/BanditsBook

Bandit algorithms help when determining the highest average reward
amongst multiple options without wasting time on bad choices.

Traditional A/B testing uses a statistical sampling of results after the
fact; this is an exploratory approach that tries different options.
Unfortunately, this approach ignores exploiting known good choices.
These algorithms balance the desire to explore for new, better options
while still earning as much reward as possible by using known good
options.

This code is derived from samples provided in Bandit Algorithms for
Website Optimization. Improvements include documentation and simplified
updates to make code more "pythonic".

Tests are included to generate test data using each of the implemented
algorithms. The data is saved in this directory as tab-separated-values.
For convenience(?), the `generate_plots.sh` script will execute the
included R scripts to provided comparative graphs of each algorithm's
performance.

A minimal R setup should install the `ggplot2` and `plyr` libraries.

.. code-block:: bash

    # Generate data and graphs
    $ ./generate_plots.sh
"""
import math
import random



def max_index(x):
    """
    Return the index of the largest item in an iterable

    :param x: Iterable of values
    :returns: Integer index
    """
    m = max(x)
    return x.index(m)


class UCB1(object):
    """
    Upper Confidence Bound algorithm (v1)

    This algorithm is *explicitly curious* about exploring new
    strategies. Each arm is executed at least once. This is important
    since the number of times the experiment is executed should be at
    least equal to the number of arms. The more times the experiment is
    executed, the better this algorithm will be able to discard
    underperforming arms.

    :param counts: List of Integer counts of the number of times an
                   option/arm has been explored
    :param values: List of Float values defining the average amount of
                   reward from each arm
    """
    def __init__(self, counts=None, values=None):
        self.counts = counts if counts is not None else []
        self.values = values if values is not None else []


    def initialize(self, n_arms):
        """
        Initialize each arm in experiment

        :param n_arms: Number of arms in experiment
        :returns: None
        """
        self.counts = [0] * n_arms
        self.values = [0.0] * n_arms


    def select_arm(self):
        """
        Select an arm number with the highest expectation for reward

        Combines the simple estimated value of each arm with a "bonus"
        quantity. This "bonus" augments the estimated value of any arm
        with a measure of how much *less* we know about that arm than
        any other arm. The effect is that we increase the effective value
        of arms we don't know much about. This encourages the algorithm
        to explore these arms even if they seem to perform a little worse
        than the best arm.
        Over time, the probability of selecting the "best" arm improve,
        but the initial runs will vary wildly.

        :returns: Arm index of the largest average reward
        """
        n_arms = len(self.counts)

        # Ensure that all arms are tried at least once
        if 0 in self.counts:
            return self.counts.index(0)

        ucb_values = [0.0] * n_arms
        total_counts = sum(self.counts)

        # Calculate a list of upper confidence bound values
        for arm in range(n_arms):
            bonus = math.sqrt((2 * math.log(total_counts)) / float(self.counts[arm]))
            ucb_values[arm] = self.values[arm] + bonus

        return max_index(ucb_values)


    def update(self, chosen_arm, reward):
        """
        Update the algorithm reward stats for the chosen arm

        :param chosen_arm: Integer of chosen arm
        :param reward: Reward value to assign to chosen arm
        """
        n = self.counts[chosen_arm] = self.counts[chosen_arm] + 1

        value = self.values[chosen_arm]
        new_value = ((n - 1) / float(n)) * value + (1 / float(n)) * reward
        self.values[chosen_arm] = new_value



class EpsilonGreedy(object):
    """
    Simple algorithm that generally exploits the best available option,
    but sometimes explores other options. Epsilon refers to the odds
    that the algorithm explores instead of exploits.

    The performance of this algorithm depends upon the epsilon value. In
    general, a high epsilon value will explore a lot to find the best
    option quickly, but it keeps exploring even after it is not worth
    doing so. In contrast, a low epsilon takes longer to explore, but
    eventually has a higher probability of selecting the best option.

    :param epsilon: Probability between 0.0 and 1.0 that the algorithm
                    will choose to *explore* available options.
                    Epsilon 0.1 means that the algorithm will explore
                    the available arms 10% of the time.
                    As Epsilon nears 1.0, it can be considered more
                    exploratory, while an Epsilon near 0 is considered
                    conservative and will exploit known good options
                    more often.
                    If None, this algorithm will gradually explore less
                    over time (anneal)
    :param counts: List of Integer counts of the number of times an
                   option/arm has been explored
    :param values: List of Float values defining the average amount of
                   reward from each arm

    """
    def __init__(self, epsilon=None, counts=None, values=None):
        self.epsilon = epsilon
        self.counts = counts if counts is not None else []
        self.values = values if values is not None else []


    def initialize(self, n_arms):
        """
        Initialize each arm in experiment

        :param n_arms: Number of arms in experiment
        :returns: None
        """
        self.counts = [0] * n_arms
        self.values = [0.0] * n_arms


    def select_arm(self):
        """
        Select an arm number with the highest expectation for reward

        The annealing algorithm will use an epsilon that gradually
        moves closer to 0 as the simulation progresses, meaning that the
        selected arms are more likely to exploit known good options
        rather than explore.

        :returns: Arm index of the largest average reward (exploit),
                  or random arm index (explore)
        """
        epsilon = self.epsilon
        if epsilon is None:
            t = sum(self.counts) + 1
            epsilon = 1 / math.log(t + 0.0000001)

        if random.random() > epsilon:
            return max_index(self.values)
        else:
            return random.randrange(len(self.values))


    def update(self, chosen_arm, reward):
        """
        Update the algorithm reward stats for the chosen arm

        :param chosen_arm: Integer of chosen arm
        :param reward: Reward value to assign to chosen arm
        """
        n = self.counts[chosen_arm] = self.counts[chosen_arm] + 1

        value = self.values[chosen_arm]
        new_value = ((n - 1) / float(n)) * value + (1 / float(n)) * reward
        self.values[chosen_arm] = new_value
