require(ggplot2)
require(scmamp)
setwd("~/Dev/mklaren/examples/rnacontext")


# Original AUCs from the original publication
proteins = c("Fusip", "VTS1", "YB1", "SLM2", "SF2", "U1A", "HuR", "PTB")
aucs = c(0.53, 0.65, 0.17, 0.81, 0.70, 0.30, 0.96, 0.69)
names(aucs) = proteins

# Read files 
in_files = sprintf("../output/rnacontext/2017-8-2/results_%s.csv", 9:17)
# in_files = sprintf("../output/rnacontext/2017-8-3/results_%s.csv", 0:8)
data = data.frame()
for (in_file in in_files){
  df = read.csv(in_file, stringsAsFactors = FALSE, header = TRUE)  
  data = rbind(data, df)
}
# ks = 1:length(unique(data$kernels))
# names(ks) = unique(data$kernels)
# data$kern = ks[data$kernels]
data$dataset = unlist(lapply(strsplit(data$dataset, "_"), function(x) x[1]))

# Cross-validation
data$id = paste(data$dataset, data$method, data$iteration, data$p, sep=".")
agg = aggregate(data$evar_va, by=list(id=data$id), max)
row.names(agg) <- agg$id
data$best = agg[data$id, "x"] == data$evar_va
data = data[data$best,]

# Average
agg.m = aggregate(data$evar, by=list(dataset=data$dataset, method=data$method), mean)
agg.s = aggregate(data$evar, by=list(dataset=data$dataset, method=data$method), sd)

# 1 v 1 matrix
datasets = unique(data$dataset)
methods = unique(data$method)
R = matrix(NA, nrow=length(datasets), ncol=length(methods))
colnames(R) <- methods
row.names(R) <- datasets
for(i in 1:nrow(agg.m)) R[agg.m[i, "dataset"], agg.m[i, "method"]] = agg.m[i, "x"]

# Store CD plot
fname = "../output/rnacontext/CD_rank_evar.pdf"
pdf(fname)
plotCD(R)
dev.off()


