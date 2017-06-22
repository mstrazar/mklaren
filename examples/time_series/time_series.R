require(ggplot2)
setwd("/Users/martin/Dev/mklaren/examples/time_series/")

# n = 1000 ; Add FITC
data = read.csv("../output/energy/2017-6-15/results_0.csv", header = TRUE)

# n = 500/500 ; Add RFF
data = read.csv("../output/energy/2017-6-20/results_2.csv", header = TRUE)

# Results for CV for different values of lambda and ranks

# Exponetial kernel
alldata = read.csv("../output/energy/2017-6-20/results_4.csv", header = TRUE)

# Matern kernel
alldata = read.csv("../output/energy/2017-6-21/results_4.csv", header = TRUE)



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
  qplot(data=data[data$rank == r,], x=as.factor(signal), 
        y=mse_y, fill=method, geom="boxplot",
        xlab="Time series", ylab="MSE", ylim=c(0, 15), 
        main=sprintf("Rank=%d", r))  
}