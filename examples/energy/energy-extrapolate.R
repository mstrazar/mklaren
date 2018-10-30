hlp = "Post-processing of experiments with 1D time series (Energy dataset)."

require(optparse)
require(ggplot2)
require(scmamp)
require(xtable)

# Defined order of methods
row.order = c("Mklaren", "CSI", "ICD", "Nystrom", "Arima")

# Number of time series to select lambda
nval = 9

# Parse input arguments
option_list = list(
  make_option(c("-i", "--input"), type="character", help="Results file (.csv)"),
  make_option(c("-o", "--output"), type="character", help="Output directory.")
);
opt_parser = OptionParser(option_list=option_list, description=hlp);
opt = parse_args(opt_parser);
in_file = opt$input
out_dir = opt$output

# Temp paths
dir.create(out_dir, showWarnings = FALSE)

# Selection of methods
alldata = read.csv(in_file, header = TRUE, stringsAsFactors = FALSE) 
methods = unique(alldata$method)
alldata$validate = (alldata$tsi <= nval)

# Select scores via cross-validation
data_val = alldata[which(alldata$validate),]
data_test = alldata[which(!alldata$validate),]

# Select optimal lambda for each configuration
af_val = aggregate(data_val$mse_y, by=list(method=data_val$method, 
                                           signal=data_val$signal, 
                                           rank=data_val$rank, 
                                           lbd=data_val$lbd), mean)
af_val = af_val[order(af_val$method, af_val$signal, af_val$x),]
af_best = af_val[!duplicated(af_val[,c("method", "signal")]),]
row.names(af_best) = sprintf("%s.%s", af_best$method, af_best$signal)

# Test prediction for optimal lambda
q = sprintf("%s.%s", data_test$method, data_test$signal)
data_test$best = af_best[q, "lbd"] == data_test$lbd
data_test = data_test[data_test$best,]

agg_mean = aggregate(data_test$mse_y, by=list(method=data_test$method, 
                                           signal=data_test$signal,
                                           rank=data_test$rank,
                                           lbd=data_test$lbd), mean)

agg_sd = aggregate(data_test$mse_y, by=list(method=data_test$method, 
                                              signal=data_test$signal,
                                              rank=data_test$rank,
                                              lbd=data_test$lbd), sd)


# Store results in tables
for (r in unique(alldata$rank)){
  k = unique(alldata$experiment)
  dr = af_best[af_best$rank == r,]
  signals = sort(unique(dr$signal))
  
  M = matrix(Inf, ncol=length(signals), nrow=length(methods))
  S = matrix(Inf, ncol=length(signals), nrow=length(methods))
  row.names(M) = methods
  colnames(M) = signals
  row.names(S) = methods
  colnames(S) = signals
  for (i in 1:nrow(agg_mean)) M[agg_mean[i,"method"], agg_mean[i,"signal"]] = agg_mean[i, "x"]
  for (i in 1:nrow(agg_sd)) S[agg_sd[i,"method"], agg_sd[i,"signal"]] = agg_sd[i, "x"]
  
  
  # Ranks 
  fname = file.path(out_dir, sprintf("cd_kernel-%s_rank-%02d.pdf", k, r))
  pdf(fname, width=8, height=5)
  plotCD(t(-M), alpha=.05)
  message(sprintf("Written %s", fname))
  
  # Friedman test on ranks
  message("Friedman test on ranks:")
  print(friedman.test(-M))
  
  # Text matrix 
  R = matrix("-", ncol=length(signals), nrow=length(row.order))
  row.names(R) = row.order
  colnames(R) = signals
  for(j in 1:ncol(M)){
    R[names(M[,j]), colnames(M)[j]] = sprintf("%.2f$\\pm$%.2f", M[,j], S[,j])
    vals = M[methods, j]
    best = names(which(vals == min(vals)))
    R[best, j] = sprintf("\\textbf{%.2f$\\pm$%.2f}", M[best,j], S[best,j])
  }
  
  # Store table
  fname = file.path(out_dir, sprintf("table_kernel-%s_rank-%02d.tex", k, r))
  tab = xtable(R)
  sink(fname)
  print(xtable(t(R)),
        sanitize.colnames.function=identity,
        sanitize.text.function=identity)
  sink()
  message(sprintf("Written %s", fname))
}