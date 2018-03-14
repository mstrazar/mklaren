require(ggplot2)
setwd("/Users/martins/Dev/mklaren/examples/risk")
fname = "output/risk_simulated/results.csv"

# Read data and plot 
data = read.csv(fname, header=TRUE, stringsAsFactors = FALSE)
qplot(data=data, x=as.factor(n), y=evar, fill=model, geom="boxplot")
qplot(data=data, x=as.factor(delta), y=evar, fill=model, geom="boxplot")

# Aggregate rankings
agg = aggregate(data$ranking, by=list(model=data$model), mean)
print(sort(agg))