require(ggplot2)

data = read.csv("../output/delve_num_kernels/2017-6-30/results_5.csv",
                header=TRUE, stringsAsFactors = FALSE)

fname = "../output/delve_num_kernels/nk_train.pdf"
qplot(data=data, x=as.factor(p), xlab="Num. kernels", y=RMSE_tr,
      ylab="RMSE (training)", geom="boxplot", fill=method)
ggsave(fname, width = 7, height = 3)

fname = "../output/delve_num_kernels/nk_test.pdf"
qplot(data=data, x=as.factor(p), xlab="Num. kernels", y=RMSE,
      ylab="RMSE (test)", geom="boxplot", fill=method)
ggsave(fname, width = 7, height = 3)

fname = "../output/delve_num_kernels/RMSE_tr_test.pdf"
pdf(fname, width = 6, height = 5)
m = max(data$RMSE_tr)
mi = min(data$RMSE_tr)
plot(c(mi, m), c(mi, m), type="l", col="black", 
     xlab="RMSE (training)", ylab="RMSE (test)")
col = rep("green", nrow(data))
col[data$method == "CSI"] = "red"
col[data$method == "Mklaren2"] = "blue"
lwd = log2(data$p)
points(data$RMSE_tr, data$RMSE, col=col, pch=20, lwd=lwd)
text(3.7, 5.0, "overfit", col="red")
dev.off()