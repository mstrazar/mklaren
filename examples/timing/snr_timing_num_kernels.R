hlp = "Post-processing of timing experiments with respect to the number of kernels."
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
num_kernels = num_kernels[num_kernels <= 316]
ranks = unique(data$rank)
ns = unique(data$n)

# Select a fixed parameter setting
for (r in ranks){
  for (n in ns){
    for (dim in dims){
      
      # Contruct table
      df = subset(data, n == n & rank == r & d == dim & p %in% num_kernels)
      methods = c(unique(df$method))
      R = matrix(Inf, nrow=length(methods), ncol=length(num_kernels))
      row.names(R) = methods
      colnames(R) = num_kernels
      for (i in 1:nrow(df)){
        j = as.character(df[i, "p"])
        R[df[i, "method"], j] = round(df[i, "time"],2)
      }

      # Reorder
      Ri = R + 0
      Ri[is.infinite(Ri)] = NA
      c1 = rowSums(is.infinite(R))                    # Number of infs
      c2 = rowSums(apply(Ri, 2, rank), na.rm = TRUE)  # Second criterion; mean time.
      inxs = order(c1, c2)
      R = R[inxs,]
      
      # Convert to chars
      Rc = R
      Rc[Rc == Inf] = "-"
      
      # Write to disk
      tab = xtable(Rc)
      align(tab) = c("r", "|", rep("r", length(num_kernels)))
      fname = file.path(out_dir, sprintf("times.num_k.%d.%d.%d.tex", dim, r, n))
      sink(fname)
      print(tab,
            sanitize.colnames.function=identity,
            sanitize.text.function=identity)
      sink()
      message(sprintf("Written %s", fname))
    }
  }
}