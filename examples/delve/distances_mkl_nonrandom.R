require(scmamp)
require(ggplot2)
setwd("~/Dev/mklaren/examples/delve")

in_dir = "../output/delve_regression/distances_nonrandom/"
in_file = file.path(in_dir, "2017-7-22/results_10.csv") # p: 7, n:1000
in_file = file.path(in_dir, "2017-7-22/results_12.csv") # p: 7, n:1000, CV
in_file = file.path(in_dir, "2017-7-22/results_15.csv") # p: 7, n:1000, rank: 20, CV
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
metric = "evar"
R[,] = 0
for(i in 1:nrow(data)) R[data[i, "dataset"], data[i, "method"]] = data[i, metric]
# fname = file.path(in_dir, sprintf("CD_%s.pdf", metric))
# pdf(fname)
plotCD(R, alpha=0.05)
# dev.off()
