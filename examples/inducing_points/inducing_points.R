hlp = "Post-processing of experiments with sampling inducing points. Draw LaTex tables with KL diveregence
    total variation and Kolmogorov-Smirnoff test statistic."

require(optparse)
require(xtable)

# Parse input arguments
option_list = list(
  make_option(c("-i", "--input"), type="character", help="Results file (.csv)"),
  make_option(c("-a", "--actives"), type="character", help="Directory with selected active sets for each run."),
  make_option(c("-o", "--output"), type="character", help="Output directory.")
);
opt_parser = OptionParser(option_list=option_list, description=hlp);
opt = parse_args(opt_parser);
in_file = opt$input
active_dir = opt$actives
out_dir = opt$output
dir.create(out_dir, showWarnings = FALSE)

# Fixed selected parameters
n = 100
lbd = 0

# Load data
# Initial results for rank 3 and rank 5 and different combinations of gamma;
# all lbd & n; wider range of noise and lambda; iterate over rank.
data = read.csv(in_file, header=TRUE, stringsAsFactors = FALSE) 
data$setting = sprintf("%s/%s",  data$sampling.model, data$noise.model)
settings = c("uniform/fixed", "uniform/increasing", "biased/fixed", "biased/increasing")

# Fill in KS test for distribution equality
for (i in 1:nrow(data)){
  row = data[i,]
  fname = sprintf("actives_method-%s_noise-%s_sampling-%s_n-%d_rank-%d_lbd-%.3f_gamma-%.3f.txt",
                   row$method, row$noise.model, row$sampling.model, row$n, row$rank, row$lbd, row$gamma)
  samp = read.table(file.path(active_dir, fname))$V1
  fname = sprintf("actives_method-%s_noise-%s_sampling-%s_n-%d_rank-%d_lbd-%.3f_gamma-%.3f.txt",
                  "True", row$noise.model, row$sampling.model, row$n, row$rank, row$lbd, row$gamma)
  tru = read.table(file.path(active_dir, fname))$V1
  test = ks.test(samp, tru, exact = FALSE)
  data[i, "KS.stat"] = test$statistic
  data[i, "KS.pvalue"] = test$p.value
}

# Dimensions
noises = unique(data$noise.model)
samplings = unique(data$sampling.model)
gammas = unique(data$gamma)
methods = unique(data$method)
ranks = c(3, 5)
metrics = c("kl.divergence", "total.variation", "KS.stat")

# Build a summary matrix for a fixed n
M = matrix("", nrow=length(settings), length(methods))
colnames(M) <- methods
row.names(M) <- settings

# Construct tables dependent on sampling metric; round at 3 decimals
grid = expand.grid(metrics, ranks, gammas, stringsAsFactors = FALSE)
colnames(grid) <- c("metric", "rank", "gamma")
for (gi in 1:nrow(grid)){
  M[,] = ""
  gr = grid[gi,]
  df = data[data$n == n & data$lbd == lbd & data$rank == gr$rank & data$gamma == gr$gamma,]
  df[,gr$metric] = round(df[,gr$metric], 3)
  
  for (i in 1:nrow(df)){
    row = df[i,]
    ri = row$setting
    ci = row$method
    score = row[,gr$metric]
    best = row[,gr$metric] == min(df[df$setting == ri, gr$metric])
    
    # Bold best scores
    sfx = ""
    if(best){
      M[ri, ci] = sprintf("\\textbf{%.3f %s}", score, sfx)  
    } else {
      M[ri, ci] = sprintf("%.3f %s", score, sfx)
    }
  }
  fname = file.path(out_dir, sprintf("%s_gamma-%.3f_rank-%d_n-%d.tex", gr$metric, gr$gamma, gr$rank, n))
  sink(fname)
  tab = xtable(M, caption = sprintf("%s; $\\gamma=%.1f$, K=%d, n=%d;", gr$metric, gr$gamma, gr$rank, n))
  align(tab) <- c("|", "r", "|", rep("l", length(methods)), "|")
  print(tab, 
        sanitize.colnames.function=identity, 
        sanitize.text.function=identity)
  sink()
  message(sprintf("Written %s", fname))
}