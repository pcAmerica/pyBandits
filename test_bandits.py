import random
import unittest

from bandits import UCB1, EpsilonGreedy


class BernoulliArm(object):
    """
    A Bernoulli arm returns 1 with probability p and 0 with
    probability 1-p

    Used to simulate binary rewards, e.g clickthroughs, signups, etc.

    If `p` is 0.2, then there is a 20% probability that a reward of 1
    (success) is returned

    :param p: Float probability between 0.0 and 1.0
    """
    def __init__(self, p):
        self.p = p

    def draw(self):
        """
        Simulate "playing" a specific arm to return a reward (or not)
        """
        if random.random() > self.p:
            return 0.0
        else:
            return 1.0



class BanditTestCase(unittest.TestCase):
    """
    Execute Monte-Carlo simulation using provided Bandit algorithms

    Note: These tests output tab-delimited results designed to generate
    graphs using R scripts in the r directory. Graphs can be created
    by executing the `generate_plots.sh` script
    """
    def make_arms(self, means=None):
        """
        Return a list of simulated arms that reward with a value of 1
        some percentage of the time and rewards with a value of 0 the
        rest.

        The default is a very simple scenario:
            4 arms output a reward 10% of the time
            1 arm (the best arm) outputs a reward 90% of the time.

        A test might want to consider other scenarios and provide
        different expected reward probabilities

        :param means: List of floats bet/ 0.0 and 1.0 representing arms
                      and the probability they will output a reward
        """
        # Init pseudorandom generator to produce same expected values
        random.seed(1)
        if means is None:
            means = [0.1, 0.2, 0.3, 0.5, 0.9]

        #random.shuffle(means)

        arms = [BernoulliArm(mu) for mu in means]
        return arms


    def _test_algorithm(self, algo, arms, num_sims, horizon):
        """
        Execute Monte-Carlo simulation using provided Bandit algorithms

        :param algo: Algorithm under test
        :param arms: An iterable of arms from which to simulate draws
        :param num_sims: Integer number of simulations to execute
        :param horizon: Integer number of times each `algo` is allowed to
                        pull on arms during each simulation
        """
        sim_nums = [0] * num_sims * horizon
        times = [0] * num_sims * horizon
        chosen_arms = [0] * num_sims * horizon
        rewards = [0.0] * num_sims * horizon
        cumulative_rewards = [0.0] * num_sims * horizon

        for sim in range(num_sims):
            sim = sim + 1
            algo.initialize(len(arms))

            for t in range(horizon):
                t = t + 1
                index = (sim - 1) * horizon + t - 1
                sim_nums[index] = sim
                times[index] = t

                chosen_arm = algo.select_arm()
                # R scripts expect arms to be 1-based
                chosen_arms[index] = chosen_arm + 1

                reward = arms[chosen_arm].draw()
                rewards[index] = reward

                if t == 1:
                    cumulative_rewards[index] = reward
                else:
                    cumulative_rewards[index] = cumulative_rewards[index - 1] + reward

                algo.update(chosen_arm, reward)

        return [sim_nums, times, chosen_arms, rewards, cumulative_rewards]


    def output_results(self, filename, results):
        with open(filename, "w") as f:
            for i in range(len(results[0])):
                f.write("\t".join([str(results[j][i]) for j in range(len(results))]) + "\n")


    def test_ucb1(self):
        """
        Test the UCB1 algorithm
        """
        arms = self.make_arms()
        algo = UCB1()
        algo.initialize(len(arms))
        results = self._test_algorithm(algo, arms, 5000, 500)
        self.output_results('ucb1_results.tsv', results)


    def test_epsilongreedy(self):
        """
        Test EpsilonGreedy algorithm
        """
        arms = self.make_arms()
        with open("epsilongreedy_results.tsv", "w") as f:

            for epsilon in [0.1, 0.2, 0.3, 0.4, 0.5]:
                algo = EpsilonGreedy(epsilon)
                algo.initialize(len(arms))
                results = self._test_algorithm(algo, arms, 5000, 500)
                key = '{0}\t'.format(epsilon)
                for i in range(len(results[0])):
                    f.write(key)
                    f.write("\t".join([str(results[j][i]) for j in range(len(results))]) + "\n")


    def test_annealing_epsilongreedy(self):
        """
        Test Annealing EpsilonGreedy algorithm
        """
        arms = self.make_arms()

        algo = EpsilonGreedy()
        algo.initialize(len(arms))
        results = self._test_algorithm(algo, arms, 5000, 500)

        self.output_results('epsilongreedy_annealing_results.tsv', results)