require(scmamp)
setwd("/Users/martin/Dev/mklaren/examples/delve/")

# testing kernels in range 1, 3, 5, ..., 100
in_dir = "2017-7-2"

# Relevant files
in_files = c(sprintf(file.path("../output/delve_regression/", in_dir, "/results_%d.csv"), 0:7))

# Range of the number of kernels (temporary)
p.range = c(0, 1, 2, 3, 5, 10, 30)

# Aggregate results
meths = c("CSI", "ICD", "Mklaren", "Nystrom",  "RFF")
datasets = c("boston", "abalone", "comp", "bank", 
             "pumadyn", "kin", "ionosphere", "census")
R = matrix(0, nrow=length(datasets), ncol=length(meths))
row.names(R) = datasets
colnames(R) = meths

for(p in p.range){
  R[,] = -1
  for (in_file in in_files){
    alldata = read.csv(in_file, 
                       header=TRUE, stringsAsFactors = FALSE)
    if(p != 0) alldata = alldata[alldata$p == p,]
    alldata$RMSE_tr[is.na(alldata$RMSE_tr)] = Inf
    alldata$RMSE_va[is.na(alldata$RMSE_va)] = Inf
    alldata$RMSE[is.na(alldata$RMSE)] = Inf
    dataset = unique(alldata$dataset)
    
    # Select lambda via cross-validation
    agg = aggregate(alldata$RMSE_va, by=list(dataset=alldata$dataset,
                                             method=alldata$method,
                                             rank=alldata$rank,
                                             iteration=alldata$iteration,
                                             p=alldata$p), min)
    row.names(agg) <- sprintf("%s.%s.%d.%s.%d", agg$dataset, agg$method, agg$rank, agg$iteration, agg$p)
    inxs = sprintf("%s.%s.%d.%s.%d", alldata$dataset, alldata$method, alldata$rank, alldata$iteration, alldata$p)
    alldata$best = alldata$RMSE_va == agg[inxs, "x"]
    data = alldata[alldata$best,]
    
    # Aggregate mean and sd per rank
    agg.m = aggregate(data[, "RMSE"], by=list(method=data[, "method"], rank=data[, "rank"]), mean)
    agg.s = aggregate(data[, "RMSE"], by=list(method=data[, "method"], rank=data[, "rank"]), sd)
    t = agg.m[agg.m$method == "L2KRR", "x"] + agg.s[agg.s$method == "L2KRR", "x"]  
    agg.valid = agg.m[agg.m$x < t,]
    agg.min = aggregate(agg.valid$rank, by=list(method=agg.valid$method), min) 
    row.names(agg.min) <- agg.min$method 
    R[dataset,meths] = agg.min[meths, "x"]
  }
  
  fname = sprintf("../output/delve_regression/rank_table_%d.tab", p)
  write.table(R, fname, row.names = FALSE, sep = "\t")
  message(sprintf("Written %s", fname))
  
  rnk = apply(R, MARGIN = 1, FUN=rank)
  
  fname = sprintf("../output/delve_regression/rank_CD_%d.pdf", p)
  pdf(fname)
  plotCD(t(rnk), alpha = 0.1)
  dev.off()
  message(sprintf("Written %s", fname))
}