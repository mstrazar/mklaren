require(ggplot)
require(scmamp)
setwd("~/Dev/mklaren/examples/string")


# Read data
# in_file = "../output/string/2017-7-31/results_4.csv"  # Same inducing set at each kernel
in_file = "../output/string/2017-7-31/results_7.csv"  # Different inducing set at each kernel
data = read.csv(in_file, stringsAsFactors = FALSE, header = TRUE)

# Cross-validation
data$id = paste(data$method, data$iteration, data$p, sep=".")
agg = aggregate(data$evar_va, by=list(id=data$id), max)
row.names(agg) <- agg$id
data$best = agg[data$id, "x"] == data$evar_va
data = data[data$best,]


# 1 v 1 matrix
data$pars = paste(data$iteration, data$p, sep=".")
pars = unique(data$pars)
methods = unique(data$method)
R = matrix(NA, nrow=length(pars), ncol=length(methods))
colnames(R) <- methods
row.names(R) <- pars
for(i in 1:nrow(data)) R[data[i, "pars"], data[i, "method"]] = data[i, "evar"]
plotCD(R)