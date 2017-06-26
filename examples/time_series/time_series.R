require(ggplot2)
setwd("/Users/martin/Dev/mklaren/examples/time_series/")

# Results for CV for different values of lambda and ranks
# Periodic
alldata = read.csv("../output/energy/2017-6-22/results_2.csv", header = TRUE)

# Exponetial kernel
alldata = read.csv("../output/energy/2017-6-22/results_0.csv", header = TRUE)
alldata = read.csv("../output/energy/2017-6-23/results_0.csv", header = TRUE) # optimize FITC

# Matern kernel
alldata = read.csv("../output/energy/2017-6-22/results_4.csv", header = TRUE)
alldata = read.csv("../output/energy/2017-6-23/results_2.csv", header = TRUE) # optimize FITC

# Select scores via cross-validation
alldata$name = sprintf("%s.%s.%s.%s", alldata$method, alldata$tsi, 
                                      alldata$signal, alldata$rank)
    # Select rows via CV
    af = aggregate(alldata$mse_val, by=list(method=alldata$method, 
                                    tsi=alldata$tsi, 
                                    signal=alldata$signal,
                                    rank=alldata$rank), min)
    row.names(af) <- sprintf("%s.%s.%s.%s", af$method, af$tsi, af$signal, af$rank)
    alldata$best = af[alldata$name, "x"] == alldata$mse_val
    data = alldata[alldata$best,]

# Validation error per each rank
for (r in unique(data$rank)){
  k = unique(data$experiment)
  fname = sprintf("../output/energy/boxplot_kernel-%s_rank-%02d.pdf", k, r)
  qplot(data=data[data$rank == r,], x=as.factor(signal), 
        y=mse_y, fill=method, geom="boxplot",
        xlab="Time series", ylab="MSE", 
        ylim=c(0, 15), 
        main=sprintf("Rank=%d kernel=%s", r, k))
  ggsave(fname, width = 10, height = 4)
  message(sprintf("Written %s", fname))
}