require(ggplot2)
setwd("/Users/martin/Dev/mklaren/examples/time_series/")

# n = 30
data = read.csv("../output/energy/2017-6-8/results_0.csv", header = TRUE)

# n = 1000
data = read.csv("../output/energy/2017-6-8/results_2.csv", header = TRUE)

x11()
qplot(data=data, x=as.factor(signal), y=mse_y, fill=method, geom="boxplot",
      xlab="Time series", ylab="MSE")

# Remember that we dont know the true function 
qplot(data=data, x=as.factor(signal), y=mse_f, fill=method, geom="boxplot",
      xlab="Time series", ylab="MSE (f)")