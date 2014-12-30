# Multi-Armed Bandits

This module implements Multi-Armed Bandit algorithms as described in
John Myles White's book,
[Bandit Algorithms for Website Optimization](http://shop.oreilly.com/product/0636920027393.do)

Bandit algorithms help when determining the highest average reward
amongst multiple options without wasting time on bad choices.

Traditional A/B testing uses a statistical sampling of results after the
fact; this is an exploratory approach that tries different options.
Unfortunately, this approach ignores exploiting known good choices.
These algorithms balance the desire to explore for new, better options
while still earning as much reward as possible by using known good
options.

This code is derived from [code samples](https://github.com/johnmyleswhite/BanditsBook)
provided in Bandit Algorithms for Website Optimization.
Improvements include documentation and simplified updates to make code
more "pythonic".

Tests are included to generate test data using each of the implemented
algorithms. The data is saved in this directory as tab-separated-values.

For convenience(?), the `generate_plots.sh` script will execute the
included R scripts to provided comparative graphs of each algorithm's
performance.

A minimal R setup should install the `ggplot2` and `plyr` libraries.

```shell
# Generate data and graphs
$ ./generate_plots.sh
```

A comparison of the accuracy of the algorithms:

![Accuracy of Different Algorithms](https://cloud.githubusercontent.com/assets/72856/5581144/36ae23ba-9012-11e4-86f5-3bd11bf693a4.png)


## Usage

How can I use this?

The only way to know whether an algorithm is suitable is to exercise it
under a variety of circumstances. Algorithms vary in performance
depending upon:

* How long you intend to run the experiment
* How many options (arms) are available
* Probabilities of reward from a particular option (arm)
* Initial expectations of the success of a particular option

The testing framework includes 3 basic tests that can be used as a
starting point to develop tests that simulate the types of rewards you
expect to see.


### What to test?

Click-through rates or user signups are good examples. These are binary
"rewards", meaning a user either clicked-through or did not, signed up,
or did not. An "arm" is an option in the software that is explored or
exploited and the reward is a successful result. If a particular arm
yields consistent rewards, a good algorithm should exploit that more
frequently than a random coin flip.

A concrete strategy might include creating a mapping of options and
assigning expected reward values. In this example, there are 2 options,
and we initialize the 'green-logo' with a value of 0.9, "seeding" the
algorithm with the expectation that the green logo performs a little
better than the unknown red logo. An exploitative algorithm will choose
the green-logo more frequently than the red (at least until/if the red
logo begins to perform better than the green).

```python
arms = {
    'red-logo': 0.0,
    'green-logo': 0.9
}
Algo(values=arms.values())
```

### How long

Each simulation selects an arm 500 times by default. You can alter this
value to see how an algorithm performs the more opportunities you provide
to select an option.

### How many options (arms) and expected rewards

The default is to consider 5 arms. 4 arms output a reward 10% of the
time, while 1 arm (the best arm) outputs a reward 90% of the time.

A test can pass a list of arm probabilities to the simulator to tweak
bot the number of arms and the relative probabilities that any specific
arm will return a reward.

### Initial expectations

The algorithms are initialized with a pessimistic expectation that all
arms are unfamiliar. The algorithm can be constructed with options that
may be initialized with higher expectations of reward for some arms