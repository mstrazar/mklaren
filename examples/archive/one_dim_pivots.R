require(ggplot2)
require(scmamp)

data = read.csv("output/one_dim_pivots/2017-4-3/05/_results.csv", header = TRUE)

# Friedman rank test ; reshape the matrix into experiments/rows form
# Warning: subsequent rows are assumed to belong to the same experiment
lvls = levels(data$method)
p = length(lvls)
dd = data

D = matrix(0, nrow=nrow(dd)/p, ncol=p)
colnames(D) = lvls
for (i in 0:nrow(D)-1){
  df = dd[(i*p):((i+1)*p - 1), ]
  D[i, df$method] = df$mse_te
}  
fname = sprintf("output/one_dim_pivots/cd.degree.pdf")
# pdf(fname)
plotCD(-D, alpha=0.05, cex=1.25)
title(sprintf("Degree: %s", d))
# dev.off()
message(sprintf("Written %s", fname))
