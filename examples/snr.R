require(ggplot2)
setwd("~/Dev/mklaren/examples")

# The methods are different whether we are allowed to use regularization or not
data = read.csv("output/snr/2017-5-17/results_5.csv", header=TRUE)  # all lbd & n; wider range of noise and lambda; iterate over rank.

# Merge results in a table for a given n (depend on noise) - store numeric format
methods = sort(as.character(unique(data$method)))
for (n in unique(data$n)){
  ranks = unique(data[data$n == n, "rank"])
  
  for (rank in ranks){
    # Select best hyperparamters for this rank and n
    df = data[data$n == n & data$rank == rank,]
    m = aggregate(df$mse_sig, by=list(method=df$method, lbd=df$lbd, noise=df$noise), FUN=mean)  # Mean performance per lambda
    mm = aggregate(m$x, by=list(method=m$method, noise=m$noise), FUN=min)                       # Best performance for all lambda
    row.names(mm) = sprintf("%s.%d", mm$method, mm$noise)
    m$best = m$x == mm[sprintf("%s.%d", m$method, m$noise), "x"]
    
    # Create a mapping for (noise level, n, rank, method) -> best lambda
    m2lbd = m[m$best, c("method", "noise",  "lbd")]
    row.names(m2lbd) = sprintf("%s.%d", mm$method, mm$noise)
    dff = df[df$lbd == m2lbd[sprintf("%s.%d", df$method, df$noise), "lbd"], ]               
    
    fname = sprintf("output/snr/images/cv_lambda_n-%d_rank-%d.pdf", n, rank)
    rp = as.numeric(round(100*(rank / n)))
    qplot(data=dff, x=as.factor(noise), y=mse_sig, geom="boxplot", fill=method,
          main=sprintf("N=%d, rank=%d %%, no. repl=%d", n, rp, max(dff$repl)+1),
          xlab="Noise", ylab="MSE")
    ggsave(fname)
  }
}