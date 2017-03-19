require(ggplot2)
data = read.csv("output/polynomial_prediction/2017-3-19/results_4.csv", header=TRUE)

# Graphical plots
qplot(data=data, x=as.factor(D), y=mse_pred, fill=method, geom="boxplot")
ggsave("output/polynomial_prediction/mse_pred_degree.pdf")
qplot(data=data, x=as.factor(D), y=mse_fit, fill=method, geom="boxplot")
ggsave("output/polynomial_prediction/mse_fit_degree.pdf")

qplot(data=data, x=as.factor(lbd), y=mse_fit, fill=method, geom="boxplot")
ggsave("output/polynomial_prediction/mse_fit_lbd.pdf")
qplot(data=data, x=as.factor(lbd), y=mse_pred, fill=method, geom="boxplot")
ggsave("output/polynomial_prediction/mse_pred_lbd.pdf")

qplot(data=data, x=as.factor(n), y=mse_fit, fill=method, geom="boxplot")
ggsave("output/polynomial_prediction/mse_fit_n.pdf")
qplot(data=data, x=as.factor(n), y=mse_pred, fill=method, geom="boxplot")
ggsave("output/polynomial_prediction/mse_pred_n.pdf")

# Wilcoxon rank.test (depending on rank)
for (target in c("mse_fit", "mse_pred")){
  rank.results = data.frame()
  for (d in c(unique(data$D), "all")){
    inxs.mkl = data$method=="Mklaren"
    inxs.csi = data$method=="CSI"
    if (d != "all"){
      inxs.mkl = inxs.mkl & data$D == as.numeric(d)
      inxs.csi = inxs.csi & data$D == as.numeric(d)
    }
    mse.mkl = data[inxs.mkl, target]
    mse.csi = data[inxs.csi, target]

    t = wilcox.test(mse.mkl, mse.csi, paired = TRUE, alternative = "less")
    mark = ""
    if(t$p.value < 0.05) mark = "*"
    if(t$p.value < 0.01) mark = "**"
    if(t$p.value < 0.001) mark = "***"

    df = data.frame(degree=d, Wp=t$p.value, mark=mark)
    rank.results = rbind(rank.results, df)
  }
  fname = sprintf("output/polynomial_prediction/wilcox.degree_%s.tab", target)
  write.table(rank.results, fname, row.names=FALSE)
  message(sprintf("Written %s", fname))
}
