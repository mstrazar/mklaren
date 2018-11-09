require(ggplot2)
setwd("~/Dev/mklaren/examples/mkl")

in_file = "results/mkl_keel_var/results.csv"
out_dir = "output/mkl_keel_var"
data = read.csv(in_file, header=TRUE, stringsAsFactors = FALSE)


# Score and sort datasets
agg = aggregate(data$snr, by=list(dataset=data$dataset), mean)
scores = agg$x
names(scores) = agg$dataset
scores = rev(sort(scores))

fname = file.path(out_dir, "datasets_snr_scores.pdf") 
pdf(fname, width = 12, height = 6)
plot(log(scores), col="white", ylim = c(-3, 20), ylab="Log SNR", xlab="Dataset")
lines(c(0, length(scores)), c(0, 0), col="red")
text(log(scores), names(scores), srt=90)
grid()
dev.off()
message(sprintf("Written %s", fname))