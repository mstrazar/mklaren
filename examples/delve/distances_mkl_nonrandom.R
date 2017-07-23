require(scmamp)
require(ggplot2)
setwd("~/Dev/mklaren/examples/delve")

in_dir = "../output/delve_regression/distances_nonrandom/"


# Changing the number of kernels
# in_file = file.path(in_dir, "2017-7-23/results_120.csv") # p: 7, n:1000, rank: 20, CV, exponential
# in_file = file.path(in_dir, "2017-7-23/results_121.csv") # p: 13, n:1000, rank: 20, CV, exponential
in_file = file.path(in_dir, "2017-7-23/results_122.csv") # p: 17, n:1000, rank: 20, CV, exponential


data = read.csv(in_file, header=TRUE, stringsAsFactors = FALSE) 
data = data[data$dataset != "ANACALT" & !is.na(data$evar),]

# Crossvalidation
agg = aggregate(data$evar_va, by=list(method=data$method, dataset=data$dataset), max)
row.names(agg) <- sprintf("%s.%s", agg$method, agg$dataset) 
inxs = sprintf("%s.%s", data$method, data$dataset)
data$best = agg[inxs, "x"] == data$evar_va
data = data[!is.na(data$dataset) & data$best,]

# Rank matrix
methods = unique(data$method)
datasets = unique(data$dataset)
R = matrix(0, ncol=length(methods), nrow=length(datasets))
row.names(R) <- datasets
colnames(R) <- methods

# Store CD plot
p = unique(data$p)
metric = "evar"
R[,] = 0
for(i in 1:nrow(data)) R[data[i, "dataset"], data[i, "method"]] = data[i, metric]
fname = file.path(in_dir, sprintf("CD_%s_%d.pdf", metric, p))
pdf(fname)
plotCD(R, alpha=0.05, main=sprintf("num. kernels = %d", p))
dev.off()
