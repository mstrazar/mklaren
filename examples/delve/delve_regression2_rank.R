require(scmamp)
require(xtable)
setwd("/Users/martin/Dev/mklaren/examples/delve/")

# testing kernels in range 1, 3, 5, ..., 100
# in_dir = "2017-7-2"

# more resolution in rank computation
in_dir = "2017-7-3"

# Relevant files
in_files = c(sprintf(file.path("../output/delve_regression/", in_dir, "/results_%d.csv"), 0:7))

# Range of the number of kernels (temporary)
p.range = c(0, 1, 2, 3, 5, 10)
rnk.checks = c(5, 10, 30)

# Aggregate results
meths = c("CSI", "ICD", "Mklaren", "Nystrom",  "RFF", "FITC", "Mklaren2")
datasets = c("boston", "abalone", "comp", "bank", 
             "pumadyn", "kin", "ionosphere", "census")
R = matrix(0, nrow=length(datasets), ncol=length(meths))
row.names(R) = datasets
colnames(R) = meths

# List of summaries
S.vec = vector(mode="list", length=length(rnk.checks)) 
names(S.vec) = as.character(rnk.checks)
  
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
    
    # Print in a table at arbitrary checkpoints
    for (r in rnk.checks){
      if(is.null(S.vec[[as.character(r)]])){
        S = matrix(0, nrow=length(datasets), ncol=length(meths))
        row.names(S) = datasets
        colnames(S) = meths
        S.vec[[as.character(r)]] = S
      } else {
        S = S.vec[[as.character(r)]]
      }
      m = agg.m[agg.m$rank == r,]
      s = agg.s[agg.s$rank == r,]
      # S[dataset, m$method] = sprintf("%1.2E$\\pm$%1.2E", m$x, s$x)
      S[dataset, m$method] = sprintf("%1.2E", m$x)
      mm = m$method[which.min(m$x)]
      S[dataset, mm] = sprintf("\\textbf{%s}", S[dataset, mm])
      S.vec[[as.character(r)]] = S
    }
    
    # Aggregate mean and sd per rank
    t = agg.m[agg.m$method == "L2KRR", "x"] + agg.s[agg.s$method == "L2KRR", "x"]  
    agg.valid = agg.m[agg.m$x < t,]
    agg.min = aggregate(agg.valid$rank, by=list(method=agg.valid$method), min) 
    row.names(agg.min) <- agg.min$method 
    R[dataset,meths] = agg.min[meths, "x"]
  }
  
  # Store checkpoints
  for (r in rnk.checks){
    S = S.vec[[as.character(r)]]
    fname = sprintf("../output/delve_regression/tex/rank_table_p-%d_rank-%d.tex", p, r)
    sink(fname)
    print(xtable(S, caption = sprintf("p=%d K=%d", p, r)), 
          sanitize.colnames.function=identity, 
          sanitize.text.function=identity)
    sink()
    message(sprintf("Written %s", fname))
  }
  
  # Store ranks
  fname = sprintf("../output/delve_regression/rank_table_%d.tab", p)
  write.table(R, fname, row.names = FALSE, sep = "\t")
  message(sprintf("Written %s", fname))
  
  # Double the rows
  rnk = apply(R, MARGIN = 1, FUN=rank)
  
  # Store CD
  fname = sprintf("../output/delve_regression/rank_CD_%d.pdf", p)
  pdf(fname)
  plotCD(t(-rnk), alpha = 0.1)
  dev.off()
  message(sprintf("Written %s", fname))
}