hlp = "Post-processing of experiments with synthetic data on string kernels. Store a CD plot."
require(optparse)
require(scmamp)

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

# Read input data
data = read.csv(in_file, stringsAsFactors = FALSE, header = TRUE)

# Cross-validation
data$id = paste(data$method, data$iteration, sep=".")
agg = aggregate(data$evar_va, by=list(id=data$id), max)
row.names(agg) <- agg$id
data$best = agg[data$id, "x"] == data$evar_va
data = data[data$best,]

# 1 v 1 matrix - RMSE
iters = unique(data$iter)
methods = unique(data$method)
R = matrix(NA, nrow=length(iters), ncol=length(methods))
colnames(R) <- methods
row.names(R) <- iters
for(i in 1:nrow(data)) R[as.character(data[i, "iteration"]), data[i, "method"]] = sqrt(data[i, "mse"])

# Store PDF
fname = file.path(out_dir, "CD_RMSE.pdf")
pdf(fname, width=4.0, height = 3.0)
plotCD(-R, alpha = 0.051)
dev.off()
message(sprintf("Written %s", fname))

print("Mean rank")
rowMeans(apply(R, 1, rank))      # Mean rank
print("Number of wins")
rowSums(apply(R, 1, rank) == 1)  # Number of wins
print("Percentage of wins")
rowMeans(apply(R, 1, rank) == 1) # Percentage of wins

print("Wilcoxon signed rank paired test")
wilcox.test(R[,"Mklaren"], R[,"CSI"], paired = TRUE, alternative = "less")