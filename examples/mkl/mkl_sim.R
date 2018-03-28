require(ggplot2)
require(reshape2)
require(scmamp)

# Paths
setwd("/Users/martins/Dev/mklaren/examples/mkl/")
in_file = "results/mkl_sim/results.csv"
outdir = "output/mkl_sim/"

# Parameters - significance
alpha = 0.05

# Read data and round
data = read.csv(in_file, header=TRUE, stringsAsFactors = FALSE)
data$noise = log10(round(data$noise, 5)) 
 
# Explained variance
fname = file.path(outdir, sprintf("evar.pdf", nam))
qplot(data=data, x=as.factor(noise), y=evar, fill=method, geom="boxplot", 
      ylab="Explained variance", xlab="Log10 noise", ylim=c(0, 1))
ggsave(fname)

# Plot signal to noise - below 0 the signals are not relevant 
fname = file.path(outdir, sprintf("snr.pdf", nam))
pdf(fname, width = 5, height = 3)
plot(data$noise, log(data$snr), xlab="Noise", ylab="log SNR")
lines(c(min(data$noise), max(data$noise)), c(0, 0), col="gray")
dev.off()
message(sprintf("Written %s", fname))

# Plot CDs
noises = unique(data$noise)
names(noises) = letters[1:length(noises)]
for(nam in names(noises) ){
  df = data[data$noise == noises[nam], c("method", "repl", "ranking", "evar")]
  dfm = dcast(data=df, formula=method~repl, fun.aggregate=mean, value.var="evar") 
  dfx = as.matrix(dfm[,2:ncol(dfm)])
  row.names(dfx) = dfm$method
  
  fname = file.path(outdir, sprintf("CD_noise_%s.pdf", nam))
  pdf(fname)
  plotCD(t(dfx), alpha=alpha)
  title(sprintf("Log10 noise: %d (p<%.2f)", noises[nam], alpha))
  dev.off()
  message(sprintf("Written %s", fname))
}

# Aggregate rankings
agg = aggregate(data$ranking, by=list(method=data$method, noise=data$noise), mean)
agg$x = round(agg$x, 2)
df = dcast(data=agg, formula=method~noise, fun.aggregate=mean, var=x)

# Write to file
fname = file.path(outdir, sprintf("ranks.tab", nam))
write.table(x=df, file=fname, row.names = FALSE, sep = "\t", quote = FALSE)
message(sprintf("Written %s", fname))