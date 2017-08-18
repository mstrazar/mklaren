require(ggplot2)
require(scmamp)
setwd("~/Dev/mklaren/examples/rnacontext")

# Original AUCs from the original publication
proteins = c("Fusip", "VTS1", "YB1", "SLM2", "SF2", "U1A", "HuR", "PTB")
aucs = c(0.53, 0.65, 0.17, 0.81, 0.70, 0.30, 0.96, 0.69)
names(aucs) = proteins

# Read files 
# in_files = sprintf("../output/rnacontext/2017-8-17/results_%d.csv", 0:8)  # set of weak sequences; n=1000
in_files = sprintf("../output/rnacontext/2017-8-18/results_%d.csv", 1:18)  # set of weak sequences; n=3000

data = data.frame()
for (in_file in in_files){
  df = read.csv(in_file, stringsAsFactors = FALSE, header = TRUE)  
  data = rbind(data, df)
}
# ks = 1:length(unique(data$kernels))
# names(ks) = unique(data$kernels)
# data$kern = ks[data$kernels]
data$dataset = unlist(lapply(strsplit(data$dataset, "_"), function(x) sprintf("%s.%s", x[1], x[4])))
data$dataset = gsub(".txt.gz", "", data$dataset)

# Cross-validation
data$id = paste(data$dataset, data$method, data$iteration, data$rank, sep=".")
agg = aggregate(data$evar_va, by=list(id=data$id), max)
row.names(agg) <- agg$id
data$best = agg[data$id, "x"] == data$evar_va
data = data[data$best,]
rank = unique(data$rank)

# Average
agg.m = aggregate(data$mse, by=list(dataset=data$dataset, method=data$method), mean)
agg.s = aggregate(data$mse, by=list(dataset=data$dataset, method=data$method), sd)

# 1 v 1 matrix
datasets = unique(data$dataset)
methods = unique(data$method)
R = matrix(NA, nrow=length(datasets), ncol=length(methods))
colnames(R) <- methods
row.names(R) <- datasets
for(i in 1:nrow(agg.m)) R[agg.m[i, "dataset"], agg.m[i, "method"]] = sqrt(agg.m[i, "x"])


# Qplot evar
fname = sprintf("../output/rnacontext/boxplot_evar_%d.pdf", rank)
qplot(data=data, x=as.factor(dataset), y=evar, geom="boxplot", fill=method, xlab="Dataset", ylab="Expl. var.")
ggsave(fname, width=10, height = 3)

# Store CD plot
fname = sprintf("../output/rnacontext/CD_rank_evar_rank_%d.pdf", rank)
pdf(fname)
plotCD(-R)
dev.off()

# Ranks
rnk = apply(R, 1, rank) 
rowSums(rnk == 1)

# Wilcoxon signed rank test
wilcox.test(R[,c("Mklaren")], R[,c("CSI")], paired = TRUE, alternative = "less")