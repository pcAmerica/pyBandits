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