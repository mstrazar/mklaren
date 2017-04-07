require(ggplot2)

# The methods are different whether we are allowed to use regularization or not
data = read.csv("output/snr/2017-4-7/results_6.csv", header=TRUE)  # lbd = 1
data = read.csv("output/snr/2017-4-7/results_7.csv", header=TRUE)  # lbd = 0

# As n grows, ICD becomes less likely to find good pivots if regularization is used.
lbd = 0
qplot(data=data[data$lbd==lbd & data$n==30,], x=as.factor(noise), y=mse_sig, fill=method, geom="boxplot", main = sprintf("lbd = %f", lbd))
qplot(data=data[data$lbd==lbd & data$n==100,], x=as.factor(noise), y=mse_sig, fill=method, geom="boxplot", main = sprintf("lbd = %f", lbd))
qplot(data=data[data$lbd==lbd & data$n==300,], x=as.factor(noise), y=mse_sig, fill=method, geom="boxplot", main = sprintf("lbd = %f", lbd))

lbd = 1
qplot(data=data[data$lbd==lbd & data$n==30,], x=as.factor(noise), y=mse_sig, fill=method, geom="boxplot", main = sprintf("lbd = %f", lbd))
qplot(data=data[data$lbd==lbd & data$n==100,], x=as.factor(noise), y=mse_sig, fill=method, geom="boxplot", main = sprintf("lbd = %f", lbd))
qplot(data=data[data$lbd==lbd & data$n==300,], x=as.factor(noise), y=mse_sig, fill=method, geom="boxplot", main = sprintf("lbd = %f", lbd))

lbd = 3
qplot(data=data[data$lbd==lbd & data$n==30,], x=as.factor(noise), y=mse_sig, fill=method, geom="boxplot", main = sprintf("lbd = %f", lbd))
qplot(data=data[data$lbd==lbd & data$n==100,], x=as.factor(noise), y=mse_sig, fill=method, geom="boxplot", main = sprintf("lbd = %f", lbd))
qplot(data=data[data$lbd==lbd & data$n==300,], x=as.factor(noise), y=mse_sig, fill=method, geom="boxplot", main = sprintf("lbd = %f", lbd))