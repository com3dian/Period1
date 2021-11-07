library(ggplot2)
library(stats)
setwd('/home/com3dian/Documents/github/Period1/MAIR/Lecture8/Group_assignment_stats/')

data <- read.csv('MAIR.csv')
colnames(data) <- c('time', 'option', 'agreement', 'gender', 'age', 'student', 'dutch', 
     'Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6', 'Q7', 'Q8', 'Q9', 'Q10')
dataClean <- data[data$age != 'dit is een test, niet gebruiken voor analyse',]
dataClean[is.na(dataClean)] <- 0

# pairwise.wilcox.test(x, g, p.adjust.method = p.adjust.methods,
#                      paired = TRUE, …)

preprocess <- function(df){
     df1<-df[, -c(1:7)]
     for (colNum in c(2, 4, 6, 8, 10)){
          df1[, colNum]= -strtoi(df1[, colNum]) + 4
     }
     return(c(rowSums(df1)))
}
dataClean$sumscore <- preprocess(dataClean)
res <- wilcox.test(sumscore ~ option, data = dataClean, p.adjust = "bonf",
                     paired = TRUE, alternative = 'greater')
res

ggplot(dataClean, aes(x = option, y = sumscore)) + 
  ggdist::stat_halfeye(
    adjust = .5, 
    width = .6, 
    .width = 0, 
    justification = -.2, 
    point_colour = NA
  ) + 
  geom_boxplot(
    width = .15, 
    outlier.shape = NA
  ) +
  ## add justified jitter from the {gghalves} package
  gghalves::geom_half_point(
    ## draw jitter on the left
    side = "l", 
    ## control range of jitter
    range_scale = .4, 
    ## add some transparency
    alpha = .3
  ) +
  coord_cartesian(xlim = c(1.2, NA), clip = "off")