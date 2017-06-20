require(ggplot2)
setwd("/Users/martin/Dev/mklaren/examples/time_series/")

# n = 1000 ; Add FITC
data = read.csv("../output/energy/2017-6-15/results_0.csv", header = TRUE)

data = read.csv("../output/energy/2017-6-20/results_2.csv", header = TRUE)

# Validation error
x11()
qplot(data=data, x=as.factor(signal), y=mse_y, fill=method, geom="boxplot",
      xlab="Time series", ylab="MSE", ylim=c(0, 15))