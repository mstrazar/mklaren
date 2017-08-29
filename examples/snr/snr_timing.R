require(xtable)
setwd("/Users/martin/Dev/mklaren/examples/snr/")

# TODO: general imputation of values
# Read data
dname = "../output/snr/timings/tex/"
in_file = "../output/snr/timings/timings.csv"
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
      
      # Impute missing values to methods
      targets = which((rowSums(is.na(R)) > 0))
      rows = which((rowSums(is.na(R)) == 0) & (rowSums(is.infinite(R)) == 0))
      for (target in targets){
        inxs = !is.na(R[target,]) & !is.infinite(R[target,])
        jnxs = is.na(R[target,])
        model = lm.fit(t(R[rows, inxs]), R[target, inxs])
        R[target, jnxs] = round(R[rows, jnxs]  %*% model$coefficients, 2)
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
      fname = file.path(dname, sprintf("times.%d.%d.%d.tex", dim, r, p))
      sink(fname)
      print(tab,
            sanitize.colnames.function=identity,
            sanitize.text.function=identity)
      sink()
      message(sprintf("Written %s", fname))
    }
  }
}