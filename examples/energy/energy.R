hlp = "Post-processing of experiments with 1D time series (Energy dataset)."

require(optparse)
require(ggplot2)
require(scmamp)
require(xtable)

# Defined order of methods
row.order = c("Mklaren", "CSI", "ICD", "Nystrom", "RFF", "RFF-NS", "SPGP", "Arima")

# Parse input arguments
option_list = list(
  make_option(c("-i", "--input"), type="character", help="Results file (.csv)"),
  make_option(c("-o", "--output"), type="character", help="Output directory.")
);
opt_parser = OptionParser(option_list=option_list, description=hlp);
opt = parse_args(opt_parser);
in_file = opt$input
out_dir = opt$output
dir.create(out_dir, showWarnings = FALSE)

# Selection of methods
alldata = read.csv(in_file, header = TRUE, stringsAsFactors = FALSE) 
methods = unique(alldata$method)


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

# Validation RMSE per each rank
for (r in unique(data$rank)){
  k = unique(data$experiment)
  fname = file.path(out_dir, sprintf("test_kernel-%s_rank-%02d.pdf", k, r))
  dr = data[data$rank == r,]
  p = qplot(data=dr, x=as.factor(signal), 
        y=sqrt(mse_y), fill=method, geom="boxplot",
        xlab="Time series", ylab="RMSE", 
        ylim=c(0, 10), 
        main=sprintf("Rank=%d kernel=%s", r, k))
  ggsave(fname, width = 10, height = 4)
  message(sprintf("Written %s", fname))
}

# Store results in tables
for (r in unique(data$rank)){
  k = unique(data$experiment)
  dr = data[data$rank == r,]

  # Convert to RMSE
  agg.m = aggregate(sqrt(dr$mse_y), by=list(method=dr$method, dataset=dr$signal), mean)
  agg.s = aggregate(sqrt(dr$mse_y), by=list(method=dr$method, dataset=dr$signal), sd)
  
  signals = sort(unique(dr$signal))
  M = matrix(Inf, ncol=length(signals), nrow=length(methods))
  S = matrix(Inf, ncol=length(signals), nrow=length(methods))
  row.names(M) = methods
  colnames(M) = signals
  row.names(S) = methods
  colnames(S) = signals
  for
  (i in 1:nrow(agg.m)) M[agg.m[i,"method"], agg.m[i,"dataset"]] = agg.m[i, "x"]
  for (i in 1:nrow(agg.s)) S[agg.s[i,"method"], agg.s[i,"dataset"]] = agg.s[i, "x"]
  
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