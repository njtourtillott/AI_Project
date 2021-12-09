library(dplyr)
cdp_list <- read.csv("cdp_2013_list.csv",header=T)
sp500 <- read.csv("sp500_list.csv",header=T)
usName <- sp500$Symbol
ticker <- paste(usName, "US", sep=" ")
graded_sp500 <- filter(cdp_list, Performance.Band != "", Ticker.Symbol %in% ticker)
write.csv(graded_sp500,"Graded_SP500_List.csv", row.names=F)