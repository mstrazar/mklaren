require(ggplot2)
require(scmamp)
setwd("~/Dev/mklaren/examples/string")

# 30 replications with K-mer plots and CV w.r.t. regularization
in_file = "../output/string_lengthscales_cv_val/2017-8-8/results_0.csv"
data = read.csv(in_file, stringsAsFactors = FALSE, header = TRUE)

# Cross-validation
data$id = paste(data$method, data$iteration, sep=".")
agg = aggregate(data$evar_va, by=list(id=data$id), max)
row.names(agg) <- agg$id
data$best = agg[data$id, "x"] == data$evar_va
data = data[data$best,]

# 1 v 1 matrix
iters = unique(data$iter)
methods = unique(data$method)
R = matrix(NA, nrow=length(iters), ncol=length(methods))
colnames(R) <- methods
row.names(R) <- iters
for(i in 1:nrow(data)) R[as.character(data[i, "iteration"]), data[i, "method"]] = round(data[i, "evar"], 2)
plotCD(R)
rowMeans(apply(-R, 1, rank))      # Mean rank
rowSums(apply(-R, 1, rank) == 1)  # Percentage of wins
rowMeans(apply(-R, 1, rank) == 1) # Number of wins

wilcox.test(R[,"Mklaren"], R[,"CSI"], paired = TRUE, alternative = "greater")