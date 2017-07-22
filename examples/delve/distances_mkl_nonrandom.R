require(scmamp)
require(ggplot2)
setwd("~/Dev/mklaren/examples/delve")

in_dir = "../output/delve_regression/distances_nonrandom/"
in_file = file.path(in_dir, "2017-7-22/results_9.csv") # p: 7, n:300
in_file = file.path(in_dir, "2017-7-22/results_10.csv") # p: 7, n:1000
data = read.csv(in_file, header=TRUE, stringsAsFactors = FALSE)

# Rank matrix
methods = unique(data$method)
datasets = unique(data$dataset)
R = matrix(0, ncol=length(methods), nrow=length(datasets))
row.names(R) <- datasets
colnames(R) <- methods

# Store CD plot
for (metric in c("evar", "corr")){
  R[,] = 0
  for(i in 1:nrow(data)) R[data[i, "dataset"], data[i, "method"]] = data[i, metric]
  fname = file.path(in_dir, sprintf("CD_%s.pdf", metric))
  pdf(fname)
  plotCD(R, alpha=0.05)
  dev.off()
}