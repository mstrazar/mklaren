require(scmamp)
require(xtable)
setwd("/Users/martin/Dev/mklaren/examples/delve/")

# Relevant files - KEEL
# name = "keel"
# in_dir = "2017-7-28"
# in_files = Sys.glob(sprintf(file.path("../output/delve_regression/", in_dir, "/results_*.csv")))

# Relevant files - DELVE
name = "delve"
in_dir = "2017-7-31"
in_files = Sys.glob(sprintf(file.path("../output/delve_regression/", in_dir, "/results_*.csv")))

# Bind rows
alldata = data.frame()
for (in_file in in_files){
  data = read.csv(in_file, header=TRUE, stringsAsFactors = FALSE)
  alldata = rbind(alldata, data)
}

# Aggregate results
meths = c("CSI", "ICD", "Mklaren", "Nystrom",  "RFF", "FITC")
datasets = unique(alldata$dataset)
R = matrix(NA, nrow=length(datasets), ncol=length(meths))
row.names(R) = datasets
colnames(R) = meths

# Store times
Time = matrix(NA, nrow=length(datasets), ncol=length(meths))
row.names(Time) = datasets
colnames(Time) = meths


# Input checks 
alldata = alldata[alldata$method == "L2KRR" | (alldata$method %in% meths),]
alldata$RMSE_tr[is.na(alldata$RMSE_tr)] = Inf
alldata$RMSE_va[is.na(alldata$RMSE_va)] = Inf
alldata$RMSE[is.na(alldata$RMSE)] = Inf

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
inxs = agg.m$method == "L2KRR"
t = agg.m[inxs, "x"] + agg.s[inxs, "x"]  
names(t) <- agg.m[inxs, "dataset"]

# Select valid entries
filt = agg.m$x < t[agg.m$dataset] & agg.m$method %in% meths
agg.valid = agg.m[filt,]
agg.min = aggregate(agg.valid$rank, by=list(dataset=agg.valid$dataset, method=agg.valid$method), min) 

# Extract minimal rank and corresponding time
for (i in 1:nrow(agg.min)){
  R[agg.min[i, "dataset"], agg.min[i, "method"]] = agg.min[i, "x"]
  t = mean(-data[data$dataset == agg.min[i, "dataset"] & 
           data$rank == agg.min[i, "x"] & 
           data$method == agg.min[i, "method"], "time"])
  Time[agg.min[i, "dataset"], agg.min[i, "method"]] = t
}
Time[is.na(Time)] = max(Time, na.rm=TRUE)

# Rank
R[is.na(R)] = 85
rnk = apply(R, MARGIN = 1, FUN=rank)
rm = sort(rowMeans(rnk))
print(rm)

# Store CD
fname = sprintf("../output/delve_regression/rank_CD_%s.pdf", name)
pdf(fname)
plotCD(-R, alpha = 0.05)
dev.off()
message(sprintf("Written %s", fname))

# Store CD
fname = sprintf("../output/delve_regression/time_CD_%s.pdf", name)
pdf(fname)
plotCD(-Time, alpha = 0.05)
dev.off()
message(sprintf("Written %s", fname))