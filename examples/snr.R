require(ggplot2)
setwd("~/Dev/mklaren/examples")

# The methods are different whether we are allowed to use regularization or not
data = read.csv("output/snr/2017-5-17/results_5.csv", header=TRUE)  # all lbd & n; wider range of noise and lambda; iterate over rank.

# Merge results in a table for a given n (depend on noise) - store numeric format
methods = sort(as.character(unique(data$method)))
noises = sort(unique(data$noise))
for (n in unique(data$n)){
  ranks = unique(data[data$n == n, "rank"])
  for (rank in ranks){
    results = data.frame(matrix(0, nrow=length(methods), ncol=length(noises)))
    row.names(results) <- methods
    colnames(results) <- sprintf("noi.%.2f", noises)
    
    # Compute a table with means
    for (noise in noises){
      col = sprintf("noi.%.2f", noise)
      df = data[data$n == n & data$rank == rank & data$noise==noise,]
      m = aggregate(df$mse_sig, by=list(method=df$method, lbd=df$lbd), FUN=mean)
      mm = aggregate(m$x, by=list(method=m$method), FUN=min)
      results[as.character(mm$method), col] = round(mm$x, 2)
    }
    
    # Store numeric data 
    fname = sprintf("output/snr/images/cv_lambda_n-%d_rank-%d.tab", n, rank)
    write.table(results, fname, row.names=TRUE)
    message(sprintf("Written %s",fname))
    
    # Select best hyperparamters for this rank and n
    df = data[data$n == n & data$rank == rank,]
    m = aggregate(df$mse_sig, by=list(method=df$method, lbd=df$lbd, noise=df$noise), FUN=mean)  # Mean performance per lambda
    mm = aggregate(m$x, by=list(method=m$method, noise=m$noise), FUN=min)                      # Best performance for all lambda
    row.names(mm) = sprintf("%s.%d", mm$method, mm$noise)
    m$best = m$x == mm[sprintf("%s.%d", m$method, m$noise), "x"]
    
    # Create a mapping for (noise level, n, rank, method) -> best lambda
    m2lbd = m[m$best, c("method", "noise",  "lbd")]
    row.names(m2lbd) = sprintf("%s.%d", mm$method, mm$noise)
    dff = df[df$lbd == m2lbd[sprintf("%s.%d", df$method, df$noise), "lbd"], ]               
    
    fname = sprintf("output/snr/images/cv_lambda_n-%d_rank-%d.pdf", n, rank)
    qplot(data=dff, x=as.factor(noise), y=mse_sig, geom="boxplot", fill=method,
          main=sprintf("N=%d, rank=%d", n, rank),
          xlab="Noise", ylab="MSE")
    ggsave(fname)
  }
}


# Merge results in a table for a given n (depend on noise) - store numeric format
# Correlation
methods = sort(as.character(unique(data$method)))
noises = sort(unique(data$noise))
for (n in unique(data$n)){
  ranks = unique(data[data$n == n, "rank"])
  for (rank in ranks){
    
    # Select best hyperparamters for this rank and n
    df = data[data$n == n & data$rank == rank,]
    m = aggregate(df$pr_rho, by=list(method=df$method, lbd=df$lbd, noise=df$noise), FUN=mean)  # Mean performance per lambda
    mm = aggregate(m$x, by=list(method=m$method, noise=m$noise), FUN=max)                      # Best performance for all lambda
    row.names(mm) = sprintf("%s.%d", mm$method, mm$noise)
    m$best = m$x == mm[sprintf("%s.%d", m$method, m$noise), "x"]
    
    # Create a mapping for (noise level, n, rank, method) -> best lambda
    m2lbd = m[m$best, c("method", "noise",  "lbd")]
    row.names(m2lbd) = sprintf("%s.%d", mm$method, mm$noise)
    dff = df[df$lbd == m2lbd[sprintf("%s.%d", df$method, df$noise), "lbd"], ]               
    
    fname = sprintf("output/snr/images/pr_lambda_n-%d_rank-%d.pdf", n, rank)
    qplot(data=dff, x=as.factor(noise), y=pr_rho, geom="boxplot", fill=method,
          main=sprintf("N=%d, rank=%d", n, rank),
          xlab="Noise", ylab="Pearson corr.")
    ggsave(fname)
  }
}