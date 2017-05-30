require(ggplot2)

# Load data
# Initial results for rank 3 and rank 5 and different combinations of gamma;
# all lbd & n; wider range of noise and lambda; iterate over rank.
data = read.csv("output/snr/2017-5-30/results_12.csv", header=TRUE)  


data$setting = sprintf("%s/%s", data$sampling.model, data$noise.model)

# Dimensions
noises = unique(data$noise.model)
samplings = unique(data$sampling.model)
gammas = unique(data$gamma)
settings = unique(data$setting)
methods = unique(data$method)


# Build a summary matrix for a fixed n
M = matrix("", nrow=length(settings), ncol=length(gammas) * length(methods))
g = expand.grid(methods, gammas)
colnames(M) <- sprintf("%s.%s", g$Var1, g$Var2)
row.names(M) <- settings

n = 100
lbd = 0
rank = 5
metric = "total.variation"
# metric = "kl.divergence"
# metric = "anchors.dist"

df = data[data$n == n & data$lbd == lbd & data$rank == rank,]
for (i in 1:nrow(df)){
  row = df[i,]
  ri = row$setting
  ci = sprintf("%s.%s", row$method, row$gamma)
  score = row[,metric]
  best = row[,metric] == min(df[df$setting == ri & df$gamma == row$gamma, metric])
  if(best){
    M[ri, ci] = sprintf("%.3f*", score)  
  } else {
    M[ri, ci] = sprintf("%.3f", score)
  }
}