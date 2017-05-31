require(ggplot2)
require(xtable)

# Output directory
setwd("/Users/martin/Dev/mklaren/examples")
tabdir = "output/snr/tables/"

# Load data
# Initial results for rank 3 and rank 5 and different combinations of gamma;
# all lbd & n; wider range of noise and lambda; iterate over rank.
data = read.csv("output/snr/2017-5-30/results_14.csv", header=TRUE, stringsAsFactors = FALSE) 
data$setting = sprintf("%s/%s",  data$sampling.model, data$noise.model)
settings = c("uniform/fixed", "uniform/increasing", "biased/fixed", "biased/increasing")

# Dimensions
noises = unique(data$noise.model)
samplings = unique(data$sampling.model)
gammas = unique(data$gamma)
methods = unique(data$method)
ranks = c(3, 5)
metrics = c("kl.divergence", "total.variation")

# Build a summary matrix for a fixed n
M = matrix("", nrow=length(settings), length(methods))
colnames(M) <- methods
row.names(M) <- settings

# Fixed parameters
n = 100
lbd = 0

# Construct tables dependent on sampling metr
grid = expand.grid(metrics, ranks, gammas, stringsAsFactors = FALSE)
colnames(grid) <- c("metric", "rank", "gamma")
for (gi in 1:nrow(grid)){
  M[,] = ""
  gr = grid[gi,]
  df = data[data$n == n & data$lbd == lbd & data$rank == gr$rank & data$gamma == gr$gamma,]
  
  for (i in 1:nrow(df)){
    row = df[i,]
    ri = row$setting
    ci = row$method
    score = row[,gr$metric]
    best = row[,gr$metric] == min(df[df$setting == ri, gr$metric])
    
    # Bold best scores
    if(best){
      M[ri, ci] = sprintf("\\textbf{%.3f}", score)  
    } else {
      M[ri, ci] = sprintf("%.3f", score)
    }
  }
  fname = file.path(tabdir, sprintf("%s_gamma-%.3f_rank-%d_n-%d.tex", gr$metric, gr$gamma, gr$rank, n))
  sink(fname)
  tab = xtable(M, caption = sprintf("%s; $\\gamma=%.1f$, K=%d, n=%d;", gr$metric, gr$gamma, gr$rank, n))
  align(tab) <- c("|", "r", "|", rep("l", length(methods)), "|")
  print(tab, 
        sanitize.colnames.function=identity, 
        sanitize.text.function=identity)
  sink()
  message(sprintf("Written %s", fname))
}
