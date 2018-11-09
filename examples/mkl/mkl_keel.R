require(ggplot2)
setwd("~/Dev/mklaren/examples/mkl")

in_file = "results/mkl_keel/results.csv"
out_dir = "output/mkl_keel"
data = read.csv(in_file, header=TRUE, stringsAsFactors = FALSE)

# Aggregate by size
agg = aggregate(data$N, by=list(dataset=data$dataset), max)