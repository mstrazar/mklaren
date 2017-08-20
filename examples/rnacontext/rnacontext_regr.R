require(ggplot2)
require(scmamp)
require(xtable)
setwd("~/Dev/mklaren/examples/rnacontext")

# Original AUCs from the original publication
proteins = c("Fusip", "VTS1", "YB1", "SLM2", "SF2", "U1A", "HuR", "PTB")
aucs = c(0.53, 0.65, 0.17, 0.81, 0.70, 0.30, 0.96, 0.69)
names(aucs) = proteins

# Read files 
in_files = sprintf("../output/rnacontext/2017-8-18/results_%d.csv", 1:18)  # set of weak sequences; n=3000

data = data.frame()
for (in_file in in_files){
  df = read.csv(in_file, stringsAsFactors = FALSE, header = TRUE)  
  data = rbind(data, df)
}
# ks = 1:length(unique(data$kernels))
# names(ks) = unique(data$kernels)
# data$kern = ks[data$kernels]
data$dataset = unlist(lapply(strsplit(data$dataset, "_"), function(x) sprintf("%s (set %s)", x[1], x[4])))
data$dataset = gsub(".txt.gz", "", data$dataset)

# Cross-validation
data$id = paste(data$dataset, data$method, data$iteration, data$rank, sep=".")
agg = aggregate(data$evar_va, by=list(id=data$id), max)
row.names(agg) <- agg$id
data$best = agg[data$id, "x"] == data$evar_va
data = data[data$best,]
rank = unique(data$rank)

# Average
agg.m = aggregate(sqrt(data$mse), by=list(dataset=data$dataset, method=data$method), mean)
agg.s = aggregate(sqrt(data$mse), by=list(dataset=data$dataset, method=data$method), sd)

# 1 v 1 matrix
datasets = unique(data$dataset)
methods = unique(data$method)
R = matrix(NA, nrow=length(datasets), ncol=length(methods))
colnames(R) <- methods
row.names(R) <- datasets
for(i in 1:nrow(agg.m)) R[agg.m[i, "dataset"], agg.m[i, "method"]] = agg.m[i, "x"]

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

# Store results in a table
M = matrix(Inf, ncol=length(datasets), nrow=length(methods))
S = matrix(Inf, ncol=length(datasets), nrow=length(methods))
row.names(M) = methods
colnames(M) = datasets
row.names(S) = methods
colnames(S) = datasets
for (i in 1:nrow(agg.m)) M[agg.m[i,"method"], agg.m[i,"dataset"]] = agg.m[i, "x"]
for (i in 1:nrow(agg.s)) S[agg.s[i,"method"], agg.s[i,"dataset"]] = agg.s[i, "x"]

# Text matrix 
Rt = matrix("", ncol=length(datasets), nrow=length(methods))
row.names(Rt) = methods
colnames(Rt) = datasets
for(j in 1:ncol(M)){
  Rt[names(M[,j]), colnames(M)[j]] = sprintf("%.3f$\\pm$%.3f", M[,j], S[,j])
  vals = round(M[methods, j], 3)
  best = names(which(vals == min(vals)))
  Rt[best, j] = sprintf("\\textbf{%.3f$\\pm$%.3f}", M[best,j], S[best,j])
}

# Append number of columns
dn = data[,c("dataset", "n")]
dn = dn[!duplicated(dn),]
row.names(dn) = dn$dataset
Rb = cbind(t(Rt), dn)
cols = c("n","Mklaren","CSI", "Nystrom", "ICD")
rows = order(Rb[,"n"])
Rb = Rb[,cols]
Rb = Rb[rows,]


# Store table
tab = xtable(Rb)
align(tab) = c("r", "r", "|", "l", "l", "l", "l")
fname = sprintf("../output/rnacontext/tex/rmse.tex")
sink(fname)
print(tab,
      sanitize.colnames.function=identity,
      sanitize.text.function=identity)
sink()
message(sprintf("Written %s", fname))