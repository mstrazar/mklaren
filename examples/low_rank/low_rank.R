hlp = "Post processing of the results on low-rank approximations
       (KEEL datasets)."

require(optparse)
require(scmamp)
require(xtable)

# Parse input arguments
option_list = list(
  make_option(c("-i", "--input"), type="character", help="Results file (.csv)"),
  make_option(c("-o", "--output"), type="character", help="Output directory")
);
opt_parser = OptionParser(option_list=option_list, description=hlp);
opt = parse_args(opt_parser);
in_file = opt$input
out_dir = opt$output

# Target methods
target = "L2KRR" 

# Low rank approximations
lowranks = c("Mklaren", "CSI", "ICD", "Nystrom", "CSI*", "ICD*",  "Nystrom*", "RFF", "RFF-NS", "SPGP")
matrices = c("Mklaren", "CSI", "ICD", "Nystrom", "CSI*", "ICD*",  "Nystrom*")
fullranks = c("L2KRR", "uniform", "Mklaren2")

# Load
alldata = read.csv(in_file, header=TRUE, stringsAsFactors = FALSE)

# Aggregate results 
meths = sort(setdiff(unique(alldata$method), fullranks))
datasets = unique(alldata$dataset)
R = matrix(NA, nrow=length(datasets), ncol=length(meths))
row.names(R) = datasets
colnames(R) = meths

# Store times
Time = matrix(NA, nrow=length(datasets), ncol=length(meths))
row.names(Time) = datasets
colnames(Time) = meths

# Select lambda via cross-validation
agg = aggregate(alldata$RMSE_va, by=list(dataset=alldata$dataset,
                                         method=alldata$method,
                                         rank=alldata$rank,
                                         iteration=alldata$iteration,
                                         p=alldata$p), min)
row.names(agg) <- sprintf("%s.%s.%d.%s.%d", agg$dataset, agg$method, agg$rank, agg$iteration, agg$p)
inxs = sprintf("%s.%s.%d.%s.%d", alldata$dataset, alldata$method, 
               alldata$rank, alldata$iteration, alldata$p)
alldata$best = alldata$RMSE_va == agg[inxs, "x"]
data = alldata[alldata$best,]

# Aggregate mean and sd per rank
agg.m = aggregate(data[, "RMSE"], by=list(dataset=data[,"dataset"], method=data[, "method"],
                    eff.rank=data[, "eff.rank"]), mean)
agg.s = aggregate(data[, "RMSE"], by=list(dataset=data[,"dataset"], method=data[, "method"],
                    eff.rank=data[, "eff.rank"]), sd)

# Aggregate mean and sd per rank
inxs = agg.m$method == target
t = agg.m[inxs, "x"] + agg.s[inxs, "x"]  
names(t) <- agg.m[inxs, "dataset"]

# Select valid entries
filt = agg.m$x < t[agg.m$dataset] & agg.m$method %in% meths
agg.valid = agg.m[filt,]
agg.min = aggregate(agg.valid$eff.rank, by=list(dataset=agg.valid$dataset, method=agg.valid$method), min) 

# Extract minimal rank and corresponding time
for (i in 1:nrow(agg.min)){
  R[agg.min[i, "dataset"], agg.min[i, "method"]] = agg.min[i, "x"]
  df = subset(data, dataset == agg.min[i, "dataset"] & 
                       rank == agg.min[i, "x"] & 
                     method == agg.min[i, "method"])
  t = mean(abs(df$time))
  Time[agg.min[i, "dataset"], agg.min[i, "method"]] = t
}

# Rank ; eliminate rows with all inifinite rank
excluded = rowSums(is.na(R[,matrices])) >= length(matrices)
message("Excluded datasets:")
print(row.names(R)[excluded])

zout = rowSums(is.na(R[,matrices])) < length(matrices)
R[is.na(R)] = Inf
R = R[zout,]
rnk = apply(R, MARGIN = 1, FUN=rank)
rm = sort(rowMeans(rnk))
message("Mean ranks:")
print(rm)


# Friedman test on ranks
message("Friedman test on ranks:")
friedman.test(R)

# Compare Mklaren and CSI
message("Wilcoxon test between Mklaren and CSI:")
print(wilcox.test(R[,c("Mklaren")], R[,c("CSI")], paired=TRUE, alternative="less"))

# Store CD
fname = file.path(out_dir, "rank_CD.pdf")
pdf(fname)
plotCD(-R, alpha = 0.05)
dev.off()
message(sprintf("Written %s", fname))

# Store CD
fname = file.path(out_dir, "rank_CD.eps")
postscript(fname)
plotCD(-R, alpha = 0.05)
dev.off()
message(sprintf("Written %s", fname))

# Dataset names to n 
dn = data[,c("dataset", "n")]
dn = dn[!duplicated(dn) & !is.na(dn$dataset),]
row.names(dn) = dn$dataset
n = dn$n
names(n) = row.names(dn)
n = n[row.names(R)]

# Store a LaTeX table
Rn = cbind(n, R[,lowranks])
Rn = Rn[order(Rn[,"n"]),]
Rt = matrix("", nrow=nrow(Rn), ncol=ncol(Rn))
colnames(Rt) = colnames(Rn)
row.names(Rt) = row.names(Rn)
for (i in 1:nrow(Rn)){
  minn = names(Rn[i,matrices])[Rn[i,matrices] == min(Rn[i,matrices])]
  Rt[i,] = Rn[i,]
  Rt[i,minn] = sprintf("\\textbf{%s}", Rn[i, minn]) 
}

# Write to output
fname = file.path(out_dir, "ranks.tex")
tab = xtable(Rt)
align(tab) = c("r", "l", "|", "|", "l", "l", "l", "l", "|", 
               "l", "l", "l", "|", "l", "l", "|", "l")
sink(fname)
print(tab, sanitize.text.function = identity)
sink()