require(ggplot2)
require(reshape2)
setwd("/Users/martins/Dev/mklaren/examples/mkl/")
fname = "results/mkl_sim/results.csv"

# Read data and round
data = read.csv(fname, header=TRUE, stringsAsFactors = FALSE)
data$evar = round(data$evar, 2)
data$noise = log10(round(data$noise, 5))

# Explained variance
qplot(data=data, x=as.factor(noise), y=evar, fill=method, geom="boxplot", ylab="evar", xlab="noise", ylim=c(0, 1))

# Ranking  
qplot(data=data, x=as.factor(noise), y=ranking, fill=method, geom="boxplot", ylab="Ranking")

# Plot signal to noise - below 0 the signals are not relevant 
plot(data$noise, log(data$snr), xlab="Noise", ylab="log SNR")
lines(c(min(data$noise), max(data$noise)), c(0, 0), color="gray")

# Aggregate rankings
agg = aggregate(data$ranking, by=list(method=data$method, noise=data$noise), mean)
dcast(data=agg, formula=method~noise, fun.aggregate=mean, var=x)