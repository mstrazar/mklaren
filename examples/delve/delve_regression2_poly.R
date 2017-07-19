require(scmamp)
require(xtable)
require(ggplot2)
setwd("/Users/martin/Dev/mklaren/examples/delve/")


in_dir = "2017-7-19"

# Relevant files
in_files = sprintf(file.path("../output/delve_regression/", in_dir, "/results_%s.csv"), 6:11)

# Aggregate results
data = data.frame()

for (in_file in in_files){
  
  # Select lambda via cross-validation
  alldata = read.csv(in_file, header=TRUE, stringsAsFactors = FALSE)
  agg = aggregate(alldata$RMSE_va, by=list(dataset=alldata$dataset,
                                           method=alldata$method,
                                           rank=alldata$rank,
                                           iteration=alldata$iteration,
                                           p=alldata$p), min)
  row.names(agg) <- sprintf("%s.%s.%d.%s.%d", agg$dataset, agg$method, agg$rank, agg$iteration, agg$p)
  inxs = sprintf("%s.%s.%d.%s.%d", alldata$dataset, alldata$method, alldata$rank, alldata$iteration, alldata$p)
  alldata$best = alldata$RMSE_va == agg[inxs, "x"]
  data = rbind(data, alldata[alldata$best,])
}

for (rank in unique(data$rank)){
  fname = file.path(sprintf("../output/delve_regression/poly_summary_%d.pdf", rank))
  qplot(data=data[data$rank == rank,], x=as.factor(dataset), y=evar, 
        xlab="Dataset", ylab = "Expl. variance",
        fill=method, geom="boxplot", ylim = c(0, 1))
  ggsave(fname, width=14, height=7)
}