# Fill in day of week
data$day = unlist(lapply(strsplit(data$date, " "), function(x) x[1]))
data$hour = unlist(lapply(strsplit(data$date, " "), function(x) x[2]))
days = data.frame(date=unique(data$day), day="")
names = c("mon", "tue", "wed", "thu", "fri", "sat", "sun")
days$day = c(rep(names, 19), names[1:5])
for (n in names){
  days[days$day == n, "week"] = 1:sum(days$day == n)
}

row.names(days) = days$date
data$dow = days[data$day, "day"]
data$week = days[data$day, "week"]


# Display data for a week
colors = c("blue", "yellow", "orange", "red")
weeks = c(17:20)
target = "T1"

x11()
plot(c(1:1008), rep(c(20), 1008), ylim=c(-0,20), col="white",
     ylab="Relative Temperature (C)", xlab="DOW")
for (wi in 1:length(weeks)){
  df = data[data$week == weeks[wi],]
  sig = df[, target]
  out = df$T_out
  z = sig - out
  lines(z, col=colors[wi])
}

write.csv(data, "/Users/martin/Dev/data/uci/energy/data.csv", row.names = FALSE, quote = FALSE)
