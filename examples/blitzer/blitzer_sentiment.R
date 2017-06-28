require(ggplot2)
require(xtable)
setwd("~/Dev/mklaren/examples/blitzer")
out_dir = "../output/blitzer_sentiment/"

# Read all datasets and stack them
# Datasets 0-3 contain true results for mklaren
# Datasets 4-7 contain true results for other methods, 
# with their own column selection methods
data = data.frame()

# Books
data1 = read.csv("../output/blitzer_sentiment/2017-6-27/results_0.csv", header = TRUE, stringsAsFactors = FALSE)
data2 = read.csv("../output/blitzer_sentiment/2017-6-27/results_4.csv", header = TRUE, stringsAsFactors = FALSE)
df  = rbind(data2, data1[data1$method == "Mklaren",])
data = rbind(data, df)

# DVD
data1 = read.csv("../output/blitzer_sentiment/2017-6-27/results_1.csv", header = TRUE, stringsAsFactors = FALSE)
data2 = read.csv("../output/blitzer_sentiment/2017-6-27/results_5.csv", header = TRUE, stringsAsFactors = FALSE)
df  = rbind(data2, data1[data1$method == "Mklaren",])
data = rbind(data, df)

# Electronics
data1 = read.csv("../output/blitzer_sentiment/2017-6-27/results_2.csv", header = TRUE, stringsAsFactors = FALSE)
data2 = read.csv("../output/blitzer_sentiment/2017-6-27/results_6.csv", header = TRUE, stringsAsFactors = FALSE)
df  = rbind(data2, data1[data1$method == "Mklaren",])
data = rbind(data, df)

# Kitchen
data1 = read.csv("../output/blitzer_sentiment/2017-6-27/results_3.csv", header = TRUE, stringsAsFactors = FALSE)
data2 = read.csv("../output/blitzer_sentiment/2017-6-27/results_7.csv", header = TRUE, stringsAsFactors = FALSE)
df  = rbind(data2, data1[data1$method == "Mklaren",])
data = rbind(data, df)


to.latex <- function(fname, df){
  # TODO: add l2krr when done
  rows = c("uniform", "align", "alignf", "alignfc", "Mklaren") 
  cols = unique(df$rank)
  M = matrix("", ncol=length(cols), nrow=length(rows))
  row.names(M) = rows
  colnames(M) = cols
  agg.m = aggregate(df$RMSE, by=list(method=df$method, rank=df$rank), mean)
  agg.s = aggregate(df$RMSE, by=list(method=df$method, rank=df$rank), sd)
  agg.best = aggregate(agg.m$x, by=list(rank=agg.m$rank), min)
  row.names(agg.best) = agg.best$rank
  agg.m$std = agg.s$x
  agg.m$best = agg.m$x == agg.best[as.character(agg.m$rank), "x"]
  row.names(agg.m) = sprintf("%s.%d", agg.m$method, agg.m$rank)
  
  for(r in rows){
    for(c in cols){
      m = agg.m[sprintf("%s.%d", r, c), "x"]
      s = agg.m[sprintf("%s.%d", r, c), "std"]
      b = agg.m[sprintf("%s.%d", r, c), "best"]
      t = sprintf("%.3f$\\pm$%.3f", m, s)
      if(b) t = sprintf("\\textbf{%s}", t)
      M[r, as.character(c)] = t
    }
  }
  sink(fname)
  print(xtable(M), 
  sanitize.colnames.function=identity, 
  sanitize.text.function=identity)
  sink()
}

# Remove uniform 
for (dset in unique(data$dataset)){
  # Plot a boxplot
  qplot(data=data[data$method != "uniform" & data$dataset == dset,], 
        x=as.factor(rank), 
        y=RMSE, fill=method, 
        geom="boxplot",
        xlab="Rank", main=dset)
  fname = file.path(out_dir, sprintf("%s.boxplot.pdf", dset))
  ggsave(fname, width = 5, height = 2)
  message(sprintf("Written %s", fname))
  
  # Store latex table
  fname = file.path(out_dir, sprintf("%s.boxplot.tex", dset))
  df = data[data$dataset == dset,]
  to.latex(fname, df)
  message(sprintf("Written %s", fname))
}