require(ggplot2)
require(scmamp)

# Increase lambda range and add noise variance
data = read.csv("output/blitzer_low_rank/2017-4-4/results_30.csv", header=TRUE) 

for (dset in unique(data$dataset)){
  dd = data[data$dataset==dset & data$degree == 1,]
  qplot(data=dd, x=as.factor(rank), fill=method, y=rmse_pred, geom="boxplot", main = dset)
  
  fname = sprintf("output/blitzer_low_rank/%s_rank_rmse.pdf", dset)
  ggsave(fname)
  message(sprintf("Written %s", fname))
}