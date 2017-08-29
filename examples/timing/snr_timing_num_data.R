hlp = "Post-processing of timing experiments with respect to the number of data points."
require(optparse)
require(xtable)

# Parse input arguments
option_list = list(
  make_option(c("-i", "--input"), type="character", help="Results file (.csv)"),
  make_option(c("-o", "--output"), type="character", help="Output directory.")
);
opt_parser = OptionParser(option_list=option_list, description=hlp);
opt = parse_args(opt_parser);
in_file = opt$input
out_dir = opt$output
dir.create(out_dir, showWarnings = FALSE)

# Read data
data = read.csv(in_file, header=TRUE, stringsAsFactors = FALSE)
dims = unique(data$d)
num_kernels = unique(data$p)
ranks = unique(data$rank)

# Select a fixed parameter setting
for (r in ranks){
  for (p in num_kernels){
    for (dim in dims){

      # Contruct table
      df = subset(data, p==p & rank == r & d == dim)
      methods = c(unique(df$method))
      ns = sort(unique(df$n))
      R = matrix(NA, nrow=length(methods), ncol=length(ns))
      row.names(R) = methods
      colnames(R) = ns
      for (i in 1:nrow(df)){
        j = as.character(df[i, "n"])
        R[df[i, "method"], j] = round(df[i, "time"],2)
      }

      # Reorder
      Ri = R + 0
      Ri[is.infinite(Ri)] = NA
      c1 = rowSums(is.infinite(R))     # Number of infs
      c2 = rowSums(Ri, na.rm = TRUE)   # Second criterion; mean time.
      inxs = order(c1, c2)
      R = R[inxs,]
      
      # Convert to chars
      Rc = R
      Rc[Rc == Inf] = "-"
      
      # Write to disk
      tab = xtable(Rc)
      align(tab) = c("r", "|", rep("r", length(ns)))
      fname = file.path(out_dir, sprintf("times.%d.%d.%d.tex", dim, r, p))
      sink(fname)
      print(tab,
            sanitize.colnames.function=identity,
            sanitize.text.function=identity)
      sink()
      message(sprintf("Written %s", fname))
    }
  }
}