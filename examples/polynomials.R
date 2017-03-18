require(ggplot2)
data = read.csv("output/polynomials/2017-3-17/results_12.csv", header=TRUE)


# Graphical plots
qplot(data=data, x=as.factor(D), y=mse, fill=method, geom="boxplot")
ggsave("output/polynomials/mse_degree.pdf")
qplot(data=data, x=as.factor(rank), y=mse, fill=method, geom="boxplot")
ggsave("output/polynomials/mse_rank.pdf")

# Wilcoxon rank.test (depending on rank)
rank.results = data.frame()
for (r in c(unique(data$rank), "all")){
  for (d in c(unique(data$D), "all")){
    inxs.mkl = data$method=="Mklaren"
    inxs.csi = data$method=="CSI"
    if (r != "all"){
      inxs.mkl = inxs.mkl & data$rank == as.numeric(r)
      inxs.csi = inxs.csi & data$rank == as.numeric(r)
    }
    if (d != "all"){
      inxs.mkl = inxs.mkl & data$D == as.numeric(d)
      inxs.csi = inxs.csi & data$D == as.numeric(d)
    }
    mse.mkl = data[inxs.mkl, "mse"]
    mse.csi = data[inxs.csi, "mse"]  
  
    t = wilcox.test(mse.mkl, mse.csi, paired = TRUE, alternative = "less")
    mark = ""
    if(t$p.value < 0.05) mark = "*"
    if(t$p.value < 0.01) mark = "**"
    if(t$p.value < 0.001) mark = "***"
    
    df = data.frame(rank=r, degree=d, Wp=t$p.value, mark=mark)
    rank.results = rbind(rank.results, df)
  }
}

# Filter and store results
results.bydegree = subset(rank.results, rank=="all")
results.byrank = subset(rank.results, degree=="all")
