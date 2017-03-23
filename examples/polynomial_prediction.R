require(ggplot2)

# Add bias to multiple kernels
data = read.csv("output/polynomial_prediction/2017-3-23/results_3.csv", header=TRUE)

# Graphical plots
qplot(data=data, x=as.factor(D), y=norm, geom="boxplot")
ggsave("output/polynomial_prediction/kernel_norm.pdf")

# MSE / degree
qplot(data=data, x=as.factor(D), y=mse_pred, fill=method, geom="boxplot")
ggsave("output/polynomial_prediction/mse_pred_degree.pdf")
qplot(data=data, x=as.factor(D), y=expl_var_pred, fill=method, geom="boxplot")
ggsave("output/polynomial_prediction/expl_var_pred_degree.pdf")

# MSE fit / degree
qplot(data=data, x=as.factor(D), y=mse_fit, fill=method, geom="boxplot")
ggsave("output/polynomial_prediction/mse_fit_degree.pdf")
qplot(data=data, x=as.factor(D), y=expl_var_fit, fill=method, geom="boxplot")
ggsave("output/polynomial_prediction/expl_var_fit_degree.pdf")

# MSE / lambda
qplot(data=data, x=as.factor(lbd), y=mse_fit, fill=method, geom="boxplot")
ggsave("output/polynomial_prediction/mse_fit_lbd.pdf")
qplot(data=data, x=as.factor(lbd), y=mse_pred, fill=method, geom="boxplot")
ggsave("output/polynomial_prediction/mse_pred_lbd.pdf")

# MSE / n
qplot(data=data, x=as.factor(n), y=mse_fit, fill=method, geom="boxplot")
ggsave("output/polynomial_prediction/mse_fit_n.pdf")
qplot(data=data, x=as.factor(n), y=mse_pred, fill=method, geom="boxplot")
ggsave("output/polynomial_prediction/mse_pred_n.pdf")

# MSE / rank
qplot(data=data, x=as.factor(rank), y=mse_fit, fill=method, geom="boxplot")
ggsave("output/polynomial_prediction/mse_fit_rank.pdf")
qplot(data=data, x=as.factor(rank), y=mse_pred, fill=method, geom="boxplot")
ggsave("output/polynomial_prediction/mse_pred_rank.pdf")

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
