hlp = "Post-processing of experiments with RBP binding affinity (RNAcontext dataset)."
require(optparse)
require(ggplot2)
require(scmamp)
require(xtable)

# Parse input arguments
option_list = list(
  make_option(c("-i", "--input"), type="character", help="Results file (.csv)"),
  make_option(c("-o", "--output"), type="character", help="Output directory.")
);
opt_parser = OptionParser(option_list=option_list, description=hlp);
opt = parse_args(opt_parser);
in_file = opt$input
out_dir = opt$output
dir.create(out_dir, showWarnings = FALSE)

# Read files
data = read.csv(in_file, stringsAsFactors = FALSE, header = TRUE)
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

# Store CD plot
fname = file.path(out_dir, "CD_rank_evar.pdf")
pdf(fname)
plotCD(-R)
dev.off()

# Ranks
rnk = apply(R, 1, rank) 
message("Number of wins:")
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
fname = file.path(out_dir, "rmse.tex")
sink(fname)
print(tab,
      sanitize.colnames.function=identity,
      sanitize.text.function=identity)
sink()
message(sprintf("Written %s", fname))