require(ggplot2)
require(scmamp)
setwd("/Users/martin/Dev/mklaren/examples/output/delve_regression/distances_cv_nonrandom/")
in_dir = "2017-7-24/"
in_files = Sys.glob(file.path(in_dir, "*.csv"))

# Load whole dataset
data = data.frame()
for (f in in_files) {
  df = read.csv(f, header=TRUE, stringsAsFactors = FALSE)
  data = rbind(data, df) 
}
data = data[(data$dataset != "ANACALT") & !is.na(data$dataset) & !is.na(data$evar_va),]

# Crossvalidation
agg = aggregate(data$evar_va, by=list(method=data$method, dataset=data$dataset, iteration=data$iteration, rank=data$rank, D=data$D), max)
row.names(agg) <- sprintf("%s.%s.%d.%d.%d", agg$method, agg$dataset, agg$iteration, agg$rank, agg$D) 
inxs = sprintf("%s.%s.%d.%d.%d", data$method, data$dataset, data$iteration, data$rank, data$D)
data$best = agg[inxs, "x"] == data$evar_va
data = data[!is.na(data$dataset) & data$best,]

# Aggregate over iterations
agg2 = aggregate(data$evar_tr, 
                 by=list(method=data$method, dataset=data$dataset, rank=data$rank, D=data$D), 
                 mean)
agg2$evar = agg2$x
data = agg2


# Ranking matrix
methods = unique(data$method)
datasets = unique(data$dataset)
R = matrix(NA, nrow=length(datasets), ncol=length(methods))
row.names(R) <- datasets
colnames(R) <- methods

# CD plots for ranks and Ds
for (D in unique(data$D)){
  for (rank in unique(data$rank)){
    R[,] = NA
    df = data[data$D == D & data$rank == rank & !is.na(data$dataset),]
    for (i in 1:nrow(df)) R[df[i, "dataset"], df[i, "method"]] = df[i, "evar"]
    
    # Filter regressible datasets
    Rd = R[rowMeans(R) > 0.3, ] 
    
    # Write to output
    fname = sprintf("CD_D-%d_rank-%d.pdf", D, rank)
    pdf(fname)
    plotCD(Rd)
    dev.off()
    message(sprintf("Written %s", fname))
  }
}