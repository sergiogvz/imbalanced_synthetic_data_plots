# Its executes other important over-sampling methods implemented in 
#  the [smotefamily](https://cran.r-project.org/web/packages/smotefamily/index.html) package of R 
install.packages('smotefamily')
install.packages("FNN")
install.packages('dbscan')
install.packages('igraph')
install.packages("ROSE")
require(smotefamily)
require(ROSE)
seedp <- 123123
div <- 4
N <- 1000
per <- 0.1

X <- read.table(paste("chess",div,"x",div,"_n",N,"_",per,".txt",sep = ""), sep=" ")

genData <- RSLS(X[,-3],X[,3])$data
genData1 <- BLSMOTE(X[,-3],X[,3])$data
genData2 <- ROSE(V3~., data=X,hmult.majo=0.1, hmult.mino=0.1, seed=seedp)$data
genData3 <- DBSMOTE(X[,-3],X[,3])$data
genData4 <- SLS(X[,-3],X[,3])$data

write.table(genData, paste("RSLS",div,"x",div,"_n",N,"_",per,".txt",sep = ""),row.names = FALSE,col.names = FALSE)
write.table(genData1, paste("BLSMOTE",div,"x",div,"_n",N,"_",per,".txt",sep = ""),row.names = FALSE,col.names = FALSE)
write.table(genData2, paste("ROSE",div,"x",div,"_n",N,"_",per,".txt",sep = ""),row.names = FALSE,col.names = FALSE)
write.table(genData3, paste("DBSMOTE",div,"x",div,"_n",N,"_",per,".txt",sep = ""),row.names = FALSE,col.names = FALSE)
write.table(genData4, paste("SLS",div,"x",div,"_n",N,"_",per,".txt",sep = ""),row.names = FALSE,col.names = FALSE)