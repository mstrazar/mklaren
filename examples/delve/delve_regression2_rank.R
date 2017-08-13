require(scmamp)
require(xtable)
setwd("/Users/martin/Dev/mklaren/examples/delve/")

# Relevant files - DELVE
# Target methods
target = "L2KRR" 

# Low rank approximations
lowranks = c("Mklaren", "CSI", "ICD", "Nystrom", "CSI*", "ICD*",  "Nystrom*", "RFF", "FITC")
matrices = c("Mklaren", "CSI", "ICD", "Nystrom", "CSI*", "ICD*",  "Nystrom*")
fullranks = c("L2KRR", "uniform", "Mklaren2")

# Load 
name = "keel"
in_file = sprintf("../output/delve_regression/results_%s.csv", name)
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
agg.m = aggregate(data[, "RMSE"], by=list(dataset=data[,"dataset"], method=data[, "method"], rank=data[, "rank"]), mean)
agg.s = aggregate(data[, "RMSE"], by=list(dataset=data[,"dataset"], method=data[, "method"], rank=data[, "rank"]), sd)

# Aggregate mean and sd per rank
inxs = agg.m$method == target
t = agg.m[inxs, "x"] + agg.s[inxs, "x"]  
names(t) <- agg.m[inxs, "dataset"]

# Select valid entries
filt = agg.m$x < t[agg.m$dataset] & agg.m$method %in% meths
agg.valid = agg.m[filt,]
agg.min = aggregate(agg.valid$rank, by=list(dataset=agg.valid$dataset, method=agg.valid$method), min) 

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
zout = rowSums(is.na(R[,matrices])) < length(matrices)
R[is.na(R)] = Inf
R = R[zout,]
rnk = apply(R, MARGIN = 1, FUN=rank)
rm = sort(rowMeans(rnk))
message("Mean ranks:")
print(rm)

# Store CD
fname = sprintf("../output/delve_regression/rank_CD_%s.pdf", name)
pdf(fname)
plotCD(-R, alpha = 0.05)
dev.off()
message(sprintf("Written %s", fname))

# Store CD
fname = sprintf("../output/delve_regression/rank_CD_%s.eps", name)
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
fname = sprintf("../output/delve_regression/tex/ranks_%s.tex", name)
tab = xtable(Rt)
align(tab) = c("r", "l", "|", "|", "l", "l", "l", "l", "|", 
               "l", "l", "l", "|", "l", "|", "l")
sink(fname)
print(tab, sanitize.text.function = identity)
sink()