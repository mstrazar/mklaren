require(scmamp)
setwd("/Users/martin/Dev/mklaren/examples/string")

in_file = "../output/string_lengthscales_cv/2017-8-7/results_2.csv" 
data = read.csv(in_file, stringsAsFactors = FALSE, header = TRUE)

# 1 v 1 matrix
iters = unique(data$iter)
methods = unique(data$method)
R = matrix(NA, nrow=length(iters), ncol=length(methods))
colnames(R) <- methods
row.names(R) <- iters
for(i in 1:nrow(data)) R[as.character(data[i, "iteration"]), data[i, "method"]] = round(data[i, "evar"], 2)
plotCD(R)
message(sprintf("Mklaren wins: %d/%d", sum(R[,"Mklaren"] > R[,"CSI"]), nrow(R)))
message("Average expl. var")
print(colMeans(R))
print(apply(R, 2, sd))

# Wilcod rank test
wilcox.test(R[,1], R[,2], paired = TRUE, alternative = "greater")