require(ggplot2)
data = read.csv("output/functions/2017-3-14/results_0.csv", header = TRUE)

p = data$prec
r = data$recall
data$f1 = 2 * p * r / (p + r)

qplot(data=data, y=kendall_rho, x=as.factor(n), fill=method, geom="boxplot")
qplot(data=data, y=pearson_rho, x=as.factor(n), fill=method, geom="boxplot")
qplot(data=data, y=spearman_rho, x=as.factor(n), fill=method, geom="boxplot")
qplot(data=data, y=prec, x=as.factor(n), fill=method, geom="boxplot")
qplot(data=data, y=f1, x=as.factor(n), fill=method, geom="boxplot")

qplot(data=data, y=recall, x=as.factor(rank), fill=method, geom="boxplot")
qplot(data=data, y=prec, x=as.factor(rank), fill=method, geom="boxplot")
qplot(data=data, y=f1, x=as.factor(rank), fill=method, geom="boxplot")

qplot(data=data, y=kendall_rho, x=as.factor(lbd), fill=method, geom="boxplot")
qplot(data=data, y=prec, x=as.factor(lbd), fill=method, geom="boxplot")
qplot(data=data, y=f1, x=as.factor(lbd), fill=method, geom="boxplot")

qplot(data=data, y=kendall_rho, x=as.factor(P), fill=method, geom="boxplot")
qplot(data=data, y=prec, x=as.factor(P), fill=method, geom="boxplot")
qplot(data=data, y=f1, x=as.factor(P), fill=method, geom="boxplot")

aggregate(data$f1, by=list(method=data$method), mean, na.rm=TRUE)
aggregate(data$prec, by=list(method=data$method), mean, na.rm=TRUE)
aggregate(data$recall, by=list(method=data$method), mean, na.rm=TRUE)