require(ggplot2)
require(scmamp)
require(Rgraphviz)
require(reshape2)

# Added RFF model  multiple kernels are tested individually.
data = read.csv("output/polynomial_overfit/2017-4-6/results_31.csv", header=TRUE)

# Situation at linear kernel
d = 1
qplot(data=data[data$D == d, ], x=log(mse_fit), y=log(mse_pred), 
      color=method, main=sprintf("Degree: %d", d))
ggsave("output/polynomial_overfit/pred_fit_degree_1.pdf")

# MSE / degree
qplot(data=data, x=as.factor(D), y=log(mse_pred), fill=method, geom="boxplot")
ggsave("output/polynomial_overfit/mse_pred_degree.pdf")

qplot(data=data, x=as.factor(D), y=log(mse_fit), fill=method, geom="boxplot")
ggsave("output/polynomial_overfit/mse_fit_degree.pdf")

# Explained variance
qplot(data=data, x=as.factor(D), y=expl_var_pred, fill=method, geom="boxplot")
qplot(data=data, x=as.factor(rank), y=expl_var_pred, fill=method, geom="boxplot")

# Break-down by rank and dimension
qplot(data=data[data$D==1,], x=as.factor(rank), y=expl_var_pred, fill=method, geom="boxplot")
qplot(data=data[data$D==2,], x=as.factor(rank), y=expl_var_pred, fill=method, geom="boxplot")
qplot(data=data[data$D==3,], x=as.factor(rank), y=expl_var_pred, fill=method, geom="boxplot")
qplot(data=data[data$D==4,], x=as.factor(rank), y=expl_var_pred, fill=method, geom="boxplot")
qplot(data=data[data$D==5,], x=as.factor(rank), y=expl_var_pred, fill=method, geom="boxplot")

target = "mse_pred"
meth = "CSI"
rank.results = data.frame()
for (d in c(unique(data$D), "all")){
  inxs.mkl = data$method=="Mklaren"
  inxs.csi = data$method==meth
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
fname = sprintf("output/polynomial_overfit/wilcox.degree_%s_%s.tab", target, meth)
write.table(rank.results, fname, row.names=FALSE)
message(sprintf("Written %s", fname))