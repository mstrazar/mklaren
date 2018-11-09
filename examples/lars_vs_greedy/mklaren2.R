hlp = "Temporary plotting script."

require(ggplot2)

in_file = "results.csv"
data = read.csv(in_file, header = TRUE, stringsAsFactors = FALSE)

# Plot norm
qplot(data=data, x=as.factor(noise), y=score, fill=method, geom="boxplot", xlab="Noise", ylab="|f-h(x)|")

# Plot correlation
qplot(data=data, x=as.factor(noise), y=corr, fill=method, geom="boxplot", xlab="Noise", ylab="Correlation")

# Plot differences and compare
for(n in unique(data$noise)){
  prm = data[data$noise == n & data$method == "lars", "corr"]
  prg = data[data$noise == n & data$method == "greedy", "corr"]
  wt = wilcox.test(prm, prg, paired = TRUE, alternative = "greater")
  message(sprintf("Noise: %.2f, win: %.2f (p=%e)", n, mean(prm>prg), wt$p.value))
  plot(prg, prm, main=sprintf("Noise: %f, win: %f", n, mean(prm>prg)))
  lines(c(-1, 1), c(-1, 1), col="gray")
}