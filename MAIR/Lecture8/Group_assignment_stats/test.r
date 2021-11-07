library(ggplot2)
library(stats)
setwd('/home/com3dian/Documents/github/Period1/MAIR/Lecture8/Group_assignment_stats/')

data <- read.csv('MAIR1.csv')
colnames(data) <- c('time', 'text_to_speech', 'agreement', 'gender', 'age', 'student', 'dutch', 
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
  for (colNum in c(1:10)){
    df1[, colNum]= strtoi(df1[, colNum]) * 2.5
  }
  return(c(rowSums(df1)))
}
dataClean$sumscore <- preprocess(dataClean)
res <- wilcox.test(sumscore ~ text_to_speech, data = dataClean, p.adjust = "bonf",
                   paired = TRUE, alternative = 'less')
res

b <- ggplot(dataClean, aes(x = text_to_speech, y = sumscore, fill = text_to_speech),  ) + 
  ggdist::stat_halfeye(
    adjust = .5, 
    width = .32, 
    .width = .1, 
    justification = -.2, 
    point_colour = NA
  ) + 
  geom_boxplot(
    width = .092, 
    outlier.shape = NA
  ) +
  ## add justified jitter from the {gghalves} package
  gghalves::geom_half_point(
    ## draw jitter on the left
    side = "l", 
    ## control range of jitter
    range_scale = .4, 
    ## add some transparency
    alpha = 0.5,
    fill = text_to_speech
  ) + 
  gghalves::geom_half_point_panel(
    aes(x = 0.5, fill = text_to_speech),
    transformation = position_jitter()
  ) + 
  coord_flip() + 
  theme(text = element_text(size=35), legend.position = c(0.18, 0.92)) + 
  scale_fill_manual(values=c('On' = "#FF6B6B", 'Off' = "#00A19D")) +
  ylab("sumscore(0-100)") + xlab("text to speech option")
b