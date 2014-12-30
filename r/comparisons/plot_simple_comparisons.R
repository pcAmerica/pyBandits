library("plyr")
library("ggplot2")

annealing.results <- read.csv("epsilongreedy_annealing_results.tsv", sep="\t", header = FALSE)
names(annealing.results) <- c("Sim", "T", "ChosenArm", "Reward", "CumulativeReward")
annealing.results <- transform(annealing.results, Algorithm = "Annealing epsilon-Greedy")

ucb1.results <- read.csv("ucb1_results.tsv", sep="\t", header = FALSE)
names(ucb1.results) <- c("Sim", "T", "ChosenArm", "Reward", "CumulativeReward")
ucb1.results <- transform(ucb1.results, Algorithm = "UCB1")

# annealing.softmax.results <- read.csv("annealing_softmax_results.csv", sep="\t", header = FALSE)
# names(annealing.softmax.results) <- c("Sim", "T", "ChosenArm", "Reward", "CumulativeReward")
# annealing.softmax.results <- transform(annealing.softmax.results, Algorithm = "Annealing Softmax")


results <- rbind(annealing.results, ucb1.results)

# Plot average reward as a function of time.
stats <- ddply(results,
               c("Algorithm", "T"),
               function (df) {mean(df$Reward)})
ggplot(stats, aes(x = T, y = V1, group = Algorithm, color = Algorithm)) +
  geom_line() +
  ylim(0, 1) +
  xlab("Time") +
  ylab("Average Reward") +
  ggtitle("Performance of Different Algorithms")
ggsave("r/graphs/simple_comparisons_average_reward.pdf")

# Plot frequency of selecting correct arm as a function of time.
# In this instance, 5 is the correct arm.
stats <- ddply(results,
               c("Algorithm", "T"),
               function (df) {mean(df$ChosenArm == 5)})
ggplot(stats, aes(x = T, y = V1, group = Algorithm, color = Algorithm)) +
  geom_line() +
  ylim(0, 1) +
  xlab("Time") +
  ylab("Probability of Selecting Best Arm") +
  ggtitle("Accuracy of Different Algorithms")
ggsave("r/graphs/simple_comparisons_average_accuracy.pdf")

# Plot variance of chosen arms as a function of time.
stats <- ddply(results,
               c("Algorithm", "T"),
               function (df) {var(df$ChosenArm)})
ggplot(stats, aes(x = T, y = V1, group = Algorithm, color = Algorithm)) +
  geom_line() +
  xlab("Time") +
  ylab("Variance of Chosen Arm") +
  ggtitle("Variability of Different Algorithms")
ggsave("r/graphs/simple_comparisons_variance_choices.pdf")

# Plot cumulative reward as a function of time.
stats <- ddply(results,
               c("Algorithm", "T"),
               function (df) {mean(df$CumulativeReward)})
ggplot(stats, aes(x = T, y = V1, group = Algorithm, color = Algorithm)) +
  geom_line() +
  xlab("Time") +
  ylab("Cumulative Reward of Chosen Arm") +
  ggtitle("Cumulative Reward of Different Algorithms")
ggsave("r/graphs/simple_comparisons_cumulative_reward.pdf")
