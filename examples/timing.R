require(ggplot2)
setwd("~/Dev/mklaren/examples")

data = read.csv("output/timing/2017-5-18/results_1.csv", stringsAsFactors = FALSE,
                header=TRUE)

vars = seq(0.5, 0.9, 0.1)

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
  ag = aggregate(df$time, by=list(method=df$method, expl_var=dv$expl_var, repl=dv$repl), FUN=min)
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