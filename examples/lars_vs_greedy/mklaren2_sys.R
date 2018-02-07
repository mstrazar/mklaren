hlp = "Temporary plotting script."

require(ggplot2)

# Read data
setwd("/Users/martins/Dev/mklaren/examples/lars_vs_greedy/")
in_file = "results2.csv"
data = read.csv(in_file, header = TRUE, stringsAsFactors = FALSE)
data = data[data$kernel == "exponential",]

# Full Wilcox test - corr
prm = data[data$method == "lars", "corr"]
prg = data[data$method == "greedy", "corr"]
stopifnot(length(prm) == length(prg))
wt = wilcox.test(prm, prg, paired = TRUE, alternative = "greater")

# Full Wilcox test - norm
prm = data[data$method == "lars", "score"]
prg = data[data$method == "greedy", "score"]
wt = wilcox.test(prm, prg, paired = TRUE, alternative = "less")

# Cartesian product
g = expand.grid(noise=unique(data$noise), 
                rank=unique(data$rank), 
                kernel=unique(data$kernel),
                p=unique(data$p))
g$pvalue = NA
g$pvalue.less = NA
g$win = NA
for(i in 1:nrow(g)){
  query = subset(data, noise==g[i,]$noise & p==g[i,]$p & rank==g[i,]$rank & kernel==g[i,]$kernel)
  pr.lars = subset(query, method=="lars")$corr
  pr.greedy = subset(query, method=="greedy")$corr
  wt = wilcox.test(pr.lars, pr.greedy, paired = TRUE, alternative = "greater")
  g[i,]$win = mean(pr.lars > pr.greedy, na.rm = TRUE)
  g[i,]$pvalue = wt$p.value

  wt.less = wilcox.test(pr.lars, pr.greedy, paired = TRUE, alternative = "less")
  g[i,]$pvalue.less = wt.less$p.value
}