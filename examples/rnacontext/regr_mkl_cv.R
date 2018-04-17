require(ggplot2)

# Process rnacontext data
setwd("/Users/martins/Dev/mklaren/examples/rnacontext/")
data = read.csv("results/regr_mkl_cv/results.csv", header=TRUE, stringsAsFactors = FALSE)
agg  = aggregate(data$ranking, by=list(method=data$method), mean)
agg

# Short name
data$dset = gsub(".txt.gz", "", unlist(lapply(data$dataset, function(x) sprintf("%s-%s", strsplit(x, "_")[[1]][1], strsplit(x, "_")[[1]][4]))))

# Write to disk
fname = "output/regr_mkl_cv/evar_cv.pdf"
qplot(data=data, x=dset, y=evar, fill=method, geom="boxplot")
ggsave(fname, width = 20, height = 8)
message(sprintf("Written %s", fname))