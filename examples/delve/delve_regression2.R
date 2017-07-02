require(ggplot2)
setwd("/Users/martin/Dev/mklaren/examples/delve/")

# Nice results of different number of kernels for p in 1-4
# in_dir = "2017-6-30"

# testing kernels in range 3, 5, ..., 100
in_dir = "2017-7-1"

in_files = c(file.path("../output/delve_regression/", in_dir, "/results_0.csv"),  # boston
             file.path("../output/delve_regression/", in_dir, "/results_1.csv"),  # abalone
             file.path("../output/delve_regression/", in_dir, "/results_2.csv"),  # bank
             file.path("../output/delve_regression/", in_dir, "/results_3.csv"),  # comp
             file.path("../output/delve_regression/", in_dir, "/results_4.csv"),  # pumadyn
             file.path("../output/delve_regression/", in_dir, "/results_5.csv"))  # kin
        

for (in_file in in_files){
  
  alldata = read.csv(in_file, 
                     header=TRUE, stringsAsFactors = FALSE)
  dataset = unique(alldata$dataset)
  dir.create(sprintf("../output/delve_regression/%s/", dataset), recursive = TRUE)
  wd = 8
  he = 5
  
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
  
  for(p in c("all", unique(data$p))){
  
    if (p == "all") inxs = 1:nrow(data)
    else inxs = data$p == as.integer(p)
    
    # Plot change in RMSE with rank.
    fname = sprintf("../output/delve_regression/%s/RMSE_test_p-%s.pdf", dataset, p)
    qplot(main=dataset, xlab="Rank", ylab="RMSE (test)",
          data=data[inxs,], x=as.factor(rank), y=RMSE, fill=method, geom="boxplot")
    ggsave(fname, width = wd, height = he)
    message(sprintf("Written %s", fname))
    
    # Potential for the methods to fit the data
    fname = sprintf("../output/delve_regression/%s/RMSE_train_p-%s.pdf", dataset, p)
    qplot(main=dataset, xlab="Rank", ylab="RMSE (training)",
          data=data[inxs,], 
          x=as.factor(rank), y=RMSE_tr, fill=method, geom="boxplot")
    ggsave(fname, width = wd, height = he)
    message(sprintf("Written %s", fname))
  }
  
  # Plot change in RMSE with rank.
  for (method in unique(alldata$method)){
    fname = sprintf("../output/delve_regression/%s/num_kernels_RMSE_test_%s.pdf", dataset, method)
    qplot(main=sprintf("%s %s lbd=0", dataset, method), 
          xlab="Rank", ylab="RMSE (test)",
          data=alldata[alldata$method == method & alldata$lambda == 0,], 
          x=as.factor(rank), y=RMSE, fill=as.factor(p), geom="boxplot")
    ggsave(fname, width = wd, height = he)
    message(sprintf("Written %s", fname))
    
    fname = sprintf("../output/delve_regression/%s/num_kernels_RMSE_train_%s.pdf", dataset, method)
    qplot(main=sprintf("%s %s lbd=0", dataset, method),  
          xlab="Rank", ylab="RMSE (training)",
          data=alldata[alldata$method == method & alldata$lambda == 0,], 
          x=as.factor(rank), y=RMSE_tr, fill=as.factor(p), geom="boxplot")
    ggsave(fname, width = wd, height = he)
    message(sprintf("Written %s", fname))
  
    fname = sprintf("../output/delve_regression/%s/lambda_freq_%s.pdf", dataset, method)
    pdf(fname, width = wd, height = he)
    hist(log10(1e-6 + data[data$method == method, "lambda"]),
         main=sprintf("Optimal lambda frequency (%s)", method), 
         xlab = "log10(lambda)")
    dev.off()
    message(sprintf("Written %s", fname))
  }

}