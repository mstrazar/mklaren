require(scmamp)
require(xtable)
require(ggplot2)
setwd("/Users/martin/Dev/mklaren/examples/delve/")

# Changed to a fixed number of kernels: 300, 1k, 10k
in_dir = "2017-7-18"

# Changed to a fixed number of kernels: 30k, 100k
# in_dir = "2017-7-19"

# Relevant files
in_files = Sys.glob(sprintf(file.path("../output/delve_regressionN/", in_dir, "/results_*.csv")))

# Aggregate results
data = data.frame()

for (in_file in in_files){
  alldata = read.csv(in_file, header=TRUE, stringsAsFactors = FALSE)
  data = rbind(data, alldata[alldata$lambda == 1e-05,])
  
  # Select lambda via cross-validation
  agg = aggregate(alldata$RMSE_va, by=list(dataset=alldata$dataset,
                                           method=alldata$method,
                                           rank=alldata$rank,
                                           iteration=alldata$iteration,
                                           p=alldata$p), min)
  row.names(agg) <- sprintf("%s.%s.%d.%s.%d", agg$dataset, agg$method, agg$rank, agg$iteration, agg$p)
  inxs = sprintf("%s.%s.%d.%s.%d", alldata$dataset, alldata$method, alldata$rank, alldata$iteration, alldata$p)
  # alldata$best = alldata$RMSE_va == agg[inxs, "x"]
  # data = rbind(data, alldata[alldata$best,])
}

ns = unique(data$n)
rank = 10
for (n in ns){
  fname = file.path(sprintf("../output/delve_regressionN/summary_%05d.pdf", n))
  qplot(data=data[data$n == n & data$rank == rank,], x=as.factor(dataset), y=evar, 
        xlab="Dataset", ylab = "Expl. variance",
        fill=method, geom="boxplot", ylim = c(0, 1))
  ggsave(fname, width=14, height=7)

  for (dset in unique(data$dataset)){
    fname = file.path(sprintf("../output/delve_regressionN/rank_%s_%05d.pdf", dset, n))
    qplot(data=data[data$n == n & data$dataset == dset,], 
          x=as.factor(rank), y=evar, 
          xlab="Dataset", ylab = "Expl. variance",
          fill=method, geom="boxplot", ylim = c(0, 1))
    ggsave(fname, width=14, height=7)
  }
}