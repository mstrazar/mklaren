require(scmamp)
wd =  "/Users/martin/Dev/mklaren/examples/output/keel/distances"
setwd(wd)
data = read.csv("results.csv", header = TRUE, stringsAsFactors = FALSE)
data = read.csv("results_1.csv", header = TRUE, stringsAsFactors = FALSE) # Different gamma ranges
# data = read.csv("results_2.csv", header = TRUE, stringsAsFactors = FALSE) # Different kernel faimilies

ranks = unique(data$rank)
datasets = unique(data$dataset)
colors = setdiff(palette(), "cyan")[1:length(ranks)]

# Effect of delta on fitting models
if(FALSE){
  for(dset in datasets){
    fname = sprintf("deltas/%s.pdf", dset)
    pdf(fname)
    rank = ranks[1]
    df = data[data$dataset==dset & data$rank == rank,]
    emax = max(data[data$dataset==dset, "evar"])
    plot(log10(df$delta), df$evar, type="l", main=dset, 
         xlab="Log delta", ylab = "Explained variance", ylim=c(0, emax), col=colors[1])
    for (ri in 2:length(ranks)){
      df = data[data$dataset==dset & data$rank == ranks[ri],]
      lines(log10(df$delta), df$evar, main=dset, 
           xlab="Log delta", ylab = "Explained variance", col=colors[ri])
    }
    grid()
    legend(1.2, 0.2, as.character(ranks), col=colors, lty=1)
    dev.off()
    message(sprintf("Written %s", fname))
  }
}


if(FALSE){
  fname = "dist_evar_corr.pdf"
  pdf(fname)
  plot(data$dist.val.corr, data$evar, xlab="Distance/value correlation", ylab="Mklaren expl. var.")
  text(0.7, 0.2, sprintf("R=%.2f", ct$estimate))
  dev.off()
  
  fname = "dist_evar_corr_names.pdf"
  pdf(fname, width = 12, height = 12)
  plot(data$dist.val.corr, data$evar, 
       xlab="Distance/value correlation", ylab="Mklaren expl. var.",
       col="white")
  text(data$dist.val.corr, data$evar, data$dataset)
  dev.off()
}


# See how gamma affects expl. var
gammas = sort(unique(data$gamma))
R = matrix(NA, nrow=length(datasets), ncol=length(gammas))
row.names(R) <- datasets
colnames(R) <- gammas
for (i in 1:nrow(data)) R[data[i,]$dataset, data[i,]$gamma] = round(data[i, "evar"], 2)
plotCD(R)



# See how gamma affects expl. var
data$tag = sprintf("%s.%d.%s",  data$dataset, data$rank, data$gamma)
kernels = sort(unique(data$kernel))
tags = sort(unique(data$tag))
R = matrix(NA, nrow=length(tags), ncol=length(kernels))
row.names(R) <- tags
colnames(R) <- kernels
for (i in 1:nrow(data)) R[data[i,]$tag, data[i,]$kernel] = round(data[i, "evar"], 2)
plotCD(R)