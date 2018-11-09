require(ggplot2)
require(reshape2)
require(scmamp)

# Paths
setwd("/Users/martins/Dev/mklaren/examples/mkl/")
in_file = "results/mkl_sim_cv/results.csv"
outdir = "output/mkl_sim_cv/"

# Parameters - significance
alpha = 0.05

# Read data and round
data = read.csv(in_file, header=TRUE, stringsAsFactors = FALSE)
data$noise = log10(round(data$noise, 5)) 

# Hyperparameters
noises = unique(data$noise)
ns = unique(data$N)
ds = unique(data$d)
names(noises) = letters[1:length(noises)]

# Explained variance
for (d in ds){
  for(N in ns){
    df = data[data$d == d & data$N == N,]
    fname = file.path(outdir, sprintf("evar_N-%03d_d-%03d.pdf", N, d))
    qplot(data=data, x=as.factor(noise), y=evar, fill=method, geom="boxplot", 
          ylab="Explained variance", xlab="Log10 noise", ylim=c(0, 1))
    ggsave(fname)
  }
}

# Plot signal to noise - below 0 the signals are not relevant 
fname = file.path(outdir, sprintf("snr.pdf"))
pdf(fname, width = 5, height = 3)
plot(data$noise, log(data$snr), xlab="Noise", ylab="log SNR")
lines(c(min(data$noise), max(data$noise)), c(0, 0), col="gray")
dev.off()
message(sprintf("Written %s", fname))

# Plot CDs
for(nam in names(noises)){
  for (d in ds){
    for(N in ns){
      df = data[data$noise == noises[nam] & data$d == d & data$N == N, c("method", "repl", "ranking", "evar")]
      if(nrow(df) == 0) next;
      dfm = dcast(data=df, formula=method~repl, fun.aggregate=mean, value.var="evar") 
      dfx = as.matrix(dfm[,2:ncol(dfm)])
      row.names(dfx) = dfm$method
      
      fname = file.path(outdir, sprintf("CD_N-%03d_d-%03d_noise-%s.pdf", N, d, nam))
      pdf(fname)
      plotCD(t(dfx), alpha=alpha)
      title(sprintf("Log10 noise: %d (p<%.2f)", noises[nam], alpha))
      dev.off()
      message(sprintf("Written %s", fname))
    }
  }
}

# Aggregate rankings
agg = aggregate(data$ranking, by=list(method=data$method, noise=data$noise), mean)
agg$x = round(agg$x, 2)
df = dcast(data=agg, formula=method~noise, fun.aggregate=mean, var=x)

# Write to file
fname = file.path(outdir, sprintf("ranks.tab", nam))
write.table(x=df, file=fname, row.names = FALSE, sep = "\t", quote = FALSE)
message(sprintf("Written %s", fname))