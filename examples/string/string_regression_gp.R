require(ggplot2)
require(scmamp)
setwd("~/Dev/mklaren/examples/string")


# Read data
# in_file = "../output/string/2017-8-1/results_0.csv"  # Different inducing set at each kernel ; 30 iters ;
# in_file = "../output/string/2017-8-1/results_1.csv"  # Different inducing set at each kernel N=500; L=500 ; 30 iters ;
in_file = "../output/string/2017-8-2/results_0.csv"  # Different inducing set at each kernel N=100; L=100 ; K in (2, 5) (N=100, L=100) 30 iters ;
data = read.csv(in_file, stringsAsFactors = FALSE, header = TRUE)

# Cross-validation
data$id = paste(data$method, data$iteration, data$rank, data$p, sep=".")
agg = aggregate(data$evar_va, by=list(id=data$id), max)
row.names(agg) <- agg$id
data$best = agg[data$id, "x"] == data$evar_va
data = data[data$best,]

# 1 v 1 matrix
data$pars = paste(data$iteration, data$p, sep=".")
pars = unique(data$pars)
methods = unique(data$method)
R = matrix(NA, nrow=length(pars), ncol=length(methods))
colnames(R) <- methods
row.names(R) <- pars
for(i in 1:nrow(data)) R[data[i, "pars"], data[i, "method"]] = round(data[i, "evar"], 2)

# Store CD plot
fname = "../output/string/CD_rank_evar.pdf"
pdf(fname)
plotCD(R)
dev.off()
