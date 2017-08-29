hlp = "Post-processing of Blitzer sentiment results. Store a LaTeX table with RMSE at different ranks,
       with corresponding boxplots (PDF)."

require(optparse)
require(ggplot2)
require(xtable)

# Parse input arguments
option_list = list(
  make_option(c("-i", "--input"), type="character", help="Results file (.csv)"),
  make_option(c("-o", "--output"), type="character", default="cbind.tab", help="Output directory")
);
opt_parser = OptionParser(option_list=option_list, description=hlp);
opt = parse_args(opt_parser);
in_file = opt$input
out_dir = opt$output

# Read data
data = read.csv(in_file, header = TRUE, stringsAsFactors = FALSE)

# Create a LaTeX table
to.latex <- function(fname, df){
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

# Plot a boxplot without uniform
dset = unique(data$dataset)
qplot(data=data, 
      x=as.factor(rank), 
      y=RMSE, fill=method, 
      geom="boxplot",
      xlab="Rank", main=dset)
fname = file.path(out_dir, sprintf("%s.boxplot.pdf", dset))
ggsave(fname, width = 5, height = 2)
message(sprintf("Written %s", fname))

# Store latex table
fname = file.path(out_dir, sprintf("%s.boxplot.tex", dset))
to.latex(fname, data)
message(sprintf("Written %s", fname))