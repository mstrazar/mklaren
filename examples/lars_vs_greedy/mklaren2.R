hlp = "Temporary plotting script."

require(ggplot2)

in_file = "mklaren2.csv"
data = read.csv(in_file, header = TRUE, stringsAsFactors = FALSE)
qplot(data=data, x=as.factor(noise), y=score, fill=method, geom="boxplot", xlab="Noise", ylab="|f-h(x)|")