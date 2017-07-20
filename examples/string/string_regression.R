require(scmamp)
require(xtable)
require(ggplot2)
require(reshape2)
setwd("/Users/martin/Dev/mklaren/examples/string/")

# Read data 
outdir = "../output/string/"
# in.file = "../output/string/2017-7-20/results_1.csv"  # p: 1-7, seed: 1-10, var: 3
in.file = "../output/string/2017-7-20/results_2.csv"  # p: 1-7, seed: 1-100, var: 3
# in.file = "../output/string/2017-7-20/results_3.csv"  # p: 1-7, seed: 1-10, var: 10
# in.file = "../output/string/2017-7-20/results_4.csv"  # p: 10, seed: 1-10, var: 10, 
data = read.csv(in.file, stringsAsFactors = FALSE, header=TRUE)

methods = unique(data$method)
p.range = unique(data$p)
iterations = unique(data$iteration)

# Fill a matrix for different number of kernels
R = matrix(0, nrow=length(iterations), ncol=length(methods))
row.names(R) <- iterations
colnames(R) <- methods
rnk.df = data.frame()
for (p in p.range){
  R[,] = 0
  for (m in methods){
    df = data[data$p == p & data$method == m,]
    R[as.character(df$iteration), m] = df$corr
  }
  rnk = apply(-R, MARGIN = 1, FUN=rank)
  for (m in methods){
    rnk.df = rbind(rnk.df, data.frame(method=m, rank=rnk[m, ], p=p))
  }
}

# Mean corellation
fname = file.path(outdir, sprintf("corr_num_kernels.pdf"))
qplot(data=data, x=as.factor(p), y=corr, geom="boxplot", fill=method,
      xlab="Num. kernels", ylab="Sp. correlation")
ggsave(fname)

# Num kernels / ranking
fname = file.path(outdir, sprintf("rank_num_kernels.pdf"))
qplot(data=rnk.df, x=as.factor(p), y=rank, geom="boxplot", fill=method,
      xlab="Num. kernels", ylab="Ranking")
ggsave(fname)

# Write average rank
agg = aggregate(rnk.df$rank, by=list(method=rnk.df$method, p=rnk.df$p), FUN=mean)
fname = file.path(outdir, sprintf("rank_num_kernels.tab"))
write.table(agg, fname, row.names=FALSE)