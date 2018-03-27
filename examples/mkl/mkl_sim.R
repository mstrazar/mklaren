require(ggplot2)
fname = "results/mkl_sim/results.csv"

data = read.csv(fname, header=TRUE, stringsAsFactors = FALSE)
data$evar = round(data$evar, 2)

qplot(data=data, x=as.factor(log(noise)), y=evar, fill=method, geom="boxplot", ylab="ranking")