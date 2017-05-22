require(ggplot2)
setwd("~/Dev/mklaren/examples")



# Small ranks with small lambda show good results for mklaren.
# Good results (no CV).
data = read.csv("output/timing/2017-5-22/results_8.csv", stringsAsFactors = FALSE,
                 header=TRUE)

# Good results (with CV for lambda).
data = read.csv("output/timing/2017-5-22/results_10.csv", stringsAsFactors = FALSE,
                header=TRUE)

# Select resuld based on cross-validation w.r.t. hyperparamters (lambda)
if ("expl_var_val" %in% colnames(data)){
  agg = aggregate(data$expl_var_val, by=list(rank=data$rank, n=data$n, repl=data$repl, method=data$method), FUN=max)
  row.names(agg) = sprintf("%d.%d.%d.%s", agg$rank, agg$n, agg$repl, agg$method)
  inxs = sprintf("%d.%d.%d.%s", data$rank, data$n, data$repl, data$method)
  data$best = agg[inxs, "x"] == data$expl_var_val
  data = data[data$best,]
}



# Plot distributions.
for (n in unique(data$n)){
  df = data[data$n == n,]
  
  # Rank vs. explained variance
  qplot(data=df, x=as.factor(rank), y=expl_var, geom="boxplot", fill=method,
        main = sprintf("N=%s", n))  
  fname = file.path(sprintf("output/timing/images/rank_evar_n-%d.pdf", n))
  ggsave(fname)
  
  # Average minimal time to reach desired expl. variance level
  # Round expl. var to nearest 10%
  dv = data[data$n == n,]
  dv$expl_var = floor(10 * dv$expl_var) / 10 
  ag = aggregate(dv$time, by=list(method=dv$method, expl_var=dv$expl_var, repl=dv$repl), FUN=min)
  qplot(data=ag, x=as.factor(expl_var), y=x, geom="boxplot", fill=method, 
        ylab="Time (s)", xlab="Explained variance threshold",
        main = sprintf("N=%s", n))
  fname = file.path(sprintf("output/timing/images/evar_time_n-%d.pdf", n))
  ggsave(fname)
  
  # Average minimal rank to reach desired expl. variance level
  ag = aggregate(df$rank, by=list(method=df$method, expl_var=dv$expl_var, repl=dv$repl), FUN=min)
  qplot(data=ag, x=as.factor(expl_var), y=x, geom="boxplot", fill=method, 
        ylab="Rank", xlab="Explained variance threshold",
        main = sprintf("N=%s", n))
  fname = file.path(sprintf("output/timing/images/evar_rank_n-%d.pdf", n))
  ggsave(fname)
}