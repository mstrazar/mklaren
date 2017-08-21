require(scmamp)
require(ggplot2)
setwd("~/Dev/mklaren/examples/delve")

# Extrapolating on different datasets
in_dir = "../output/delve_regression/distances_nonrandom/"

# Load whole dataset
in_dir = "../output/delve_regression/distances_nonrandom/2017-7-28/" # 3000 samples
in_files = Sys.glob(file.path(in_dir, "*.csv"))
data = data.frame()
for (f in in_files) {
  df = read.csv(f, header=TRUE, stringsAsFactors = FALSE)
  data = rbind(data, df) 
}
data = data[(data$dataset != "ANACALT") & !is.na(data$dataset) & !is.na(data$evar_va),]




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
for (metric in c("evar", "corr")){
  R[,] = NA
  for(i in 1:nrow(data)) R[data[i, "dataset"], data[i, "method"]] = round(data[i, metric], 2)
  Rd = R[rowMeans(R) > 0.2, ]
  fname = file.path(in_dir, sprintf("CD_%s_%d.pdf", metric, p))
  pdf(fname)
  plotCD(R, alpha=0.05, main=sprintf("num. kernels = %d", p))
  dev.off()
  message(sprintf("Written %s", fname))
}