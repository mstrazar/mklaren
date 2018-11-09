# Process rnacontext data
setwd("/Users/martins/Dev/mklaren/examples/rnacontext/")
data = read.csv("results/regr_mkl/results.csv", header=TRUE, stringsAsFactors = FALSE)
agg  = aggregate(data$ranking, by=list(method=data$method), mean)
agg