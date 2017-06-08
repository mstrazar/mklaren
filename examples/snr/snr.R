require(ggplot2)
setwd("~/Dev/mklaren/examples")

# The methods are different whether we are allowed to use regularization or not
data = read.csv("output/snr/2017-5-17/results_5.csv", header=TRUE)  # all lbd & n; wider range of noise and lambda; iterate over rank.
data = read.csv("output/snr/2017-5-18/results_1.csv", header=TRUE)  # all lbd & n; wider range of noise and lambda; iterate over rank.

data = read.csv("output/snr/2017-5-21/results_1.csv", header=TRUE)  # all lbd & n; wider range of noise and lambda; iterate over rank.
data = data[data$repl < 2,]



# Merge results in a table for a given n (depend on noise) - store numeric format
methods = sort(as.character(unique(data$method)))

for (n in unique(data$n)){
  ranks = unique(data[data$n == n, "rank"])
  
  for (rank in ranks){
    # Select best hyperparamters for this rank and n
    df = data[data$n == n & data$rank == rank,]
    m = aggregate(df$mse_sig, by=list(method=df$method, lbd=df$lbd, noise=df$noise), FUN=mean)  # Mean performance per lambda
    mm = aggregate(m$x, by=list(method=m$method, noise=m$noise), FUN=min)                       # Best performance for all lambda
    row.names(mm) = sprintf("%s.%f", mm$method, mm$noise)
    m$best = m$x == mm[sprintf("%s.%f", m$method, m$noise), "x"]
    
    # Create a mapping for (noise level, n, rank, method) -> best lambda
    m2lbd = m[m$best, c("method", "noise",  "lbd")]
    row.names(m2lbd) = sprintf("%s.%f", m2lbd$method, m2lbd$noise)
    dff = df[df$lbd == m2lbd[sprintf("%s.%f", df$method, df$noise), "lbd"], ]               
    
    rp = as.numeric(round(100*(rank / n)))
    fname = sprintf("output/snr/images/cv_lambda_n-%d_rank-%d.pdf", n, rank)
    qplot(data=dff, x=as.factor(noise), y=mse_sig, geom="boxplot", fill=method,
          main=sprintf("N=%d, gamma=%0.2f, rank=%d %%, no. repl=%d", n, gam, rp, max(dff$repl)+1),
          xlab="Noise", ylab="MSE")
    ggsave(fname)
    
    fname = sprintf("output/snr/images/cv_rmse_lambda_n-%d_rank-%d.pdf", n, rank)
    qplot(data=dff, x=as.factor(noise), y=sqrt(mse_sig), geom="boxplot", fill=method,
          main=sprintf("N=%d, gamma=%0.2f, rank=%d %%, no. repl=%d", n, gam, rp, max(dff$repl)+1),
          xlab="Noise", ylab="RMSE (%)")
    ggsave(fname)
    
    fname = sprintf("output/snr/images/cv_pr_lambda_n-%d_rank-%d.pdf", n, rank)
    qplot(data=dff, x=as.factor(noise), y=pr_rho, geom="boxplot", fill=method,
          main=sprintf("N=%d, gamma=%0.2f, rank=%d %%, no. repl=%d", n, gam, rp, max(dff$repl)+1),
          xlab="Noise", ylab="Pearson rho")
    ggsave(fname)
  }
}


# Effect independent of lambda
for (n in unique(data$n)){
  for (rank in unique(data[data$n == n, "rank"])){
    df = data[data$lbd == 0 & data$n == n & data$rank == rank, ]  
    fname = sprintf("output/snr/images/plain_rmse_n-%d_rank-%d.pdf", n, rank)
    qplot(data=df, x=as.factor(noise), y=sqrt(mse_sig), geom="boxplot", fill=method,
          main=sprintf("N=%d, rank=%d %%, no. repl=%d", n, rp, max(dff$repl)+1),
          xlab="Noise", ylab="RMSE")
    ggsave(fname)
  }
}
