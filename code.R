

#Install all necessary packages - TTR, neuralnet and ggplot2

library(TTR)
library(neuralnet)
library(ggplot2)
require(lubridate)  
library(plyr)  


#Reading the data.
data <- read.csv("data_SP.csv")


#Preprocessing
data$Date <- as.Date(data$Date,format = "%Y-%m-%d")

#For converting it into a timeseries data
ts <- ts(data)

#Flipping the data in ascending order
data1 = data[order(nrow(data):1),] 

#Adding columns which contains past data (1 to 5 days)

data1 <- data1[!is.na(data1$Close),]
length=nrow(data1)
data1$Price = c(data1$Close)
data1$Price_1 = c(0,head(data1$Close,-1))
data1$Price_2 = c(0,0,head(data1$Close,-2))
data1$Price_3 = c(0,0,0,head(data1$Close,-3))
data1$Price_4 = c(0,0,0,0,head(data1$Close,-4))
data1$Price_5 = c(0,0,0,0,0,head(data1$Close,-5))

#Calculate simple moving average for 5, 10 and 50 days

data1$SMA_5=SMA(data1$Close,n=5)    
data1$SMA_10=SMA(data1$Close,n=10)
data1$SMA_50=SMA(data1$Close,n=50)

#Assignming 0 to NAs

data1$SMA_5[is.na(data1$SMA_5)]<-0
data1$SMA_10[is.na(data1$SMA_10)]<-0
data1$SMA_50[is.na(data1$SMA_50)]<-0

close=data.frame(data1$Close)
data1$tr=c(close[2:length,],0)
data1$tr2 = c(close[3:length,],0,0)
data1$tr3 = c(close[4:length,],0,0,0)
data1$tr4 = c(close[5:length,],0,0,0,0)
data1$tr5 = c(close[6:length,],0,0,0,0,0)

write.csv(data1,"datafinal_model.csv")
datanew=read.table("datafinal_model.csv",header=TRUE,sep=",")
datanew=datanew[-c(1:50,length, length-1, length-2, length-3, length-4),];
data2=datanew[,9:22]

#Normalising the data

norm.fun = function(x){
  (x - min(x))/(max(x) - min(x)) 
}

#Applying norm function from column 4 to column 21 in both train and validation data
data3=apply(data2[,(2:ncol(data2))],2,norm.fun)

#70% of observations is training data. Computing its length
train.length=round(length*0.70) 

train.data=data3[1:train.length,] #Storing Traindata
val.data=data3[-c(1:train.length),-c(9,10,11,12,13)] #Remaining 30% is Validation data 

#Variables used : Past 5 days prices 

#Modelling

infy831<- neuralnet(tr + tr2 + tr3 + tr4 + tr5 ~ Price_1 + Price_2 + Price_3 +
                      Price_4 + Price_5 + SMA_5 + SMA_10 + SMA_50,
                    train.data, 
                    hidden = c(6), threshold = 0.1,
                    stepmax = 10000, rep = 10,algorithm = "rprop+",
                    err.fct = "sse", 
                    linear.output = TRUE, exclude = NULL,
                    constant.weights = NULL, likelihood = FALSE)
#Plot the model
plot(infy831, rep = "best");

pred= compute(infy831, val.data,rep=1)
result = data.frame(actual = data3[-c(1:train.length),c(9,10,11,12,13)], 
                    prediction = pred$net.result)

ggplot(result, aes(1:nrow(val.data))) +  # basic graphical object
  geom_line(aes(y=result$actual.tr), colour="green") +  # first layer
  geom_line(aes(y=result$prediction.1), colour="red")  # second layer

ggplot(result, aes(1:nrow(val.data))) +  # basic graphical object
  geom_line(aes(y=result$actual.tr2), colour="green") +  # first layer
  geom_line(aes(y=result$prediction.2), colour="red")  # second layer

ggplot(result, aes(1:nrow(val.data))) +  # basic graphical object
  geom_line(aes(y=result$actual.tr3), colour="green") +  # first layer
  geom_line(aes(y=result$prediction.3), colour="red")  # second layer

ggplot(result, aes(1:nrow(val.data))) +  # basic graphical object
  geom_line(aes(y=result$actual.tr4), colour="green") +  # first layer
  geom_line(aes(y=result$prediction.4), colour="red")  # second layer

ggplot(result, aes(1:nrow(val.data))) +  # basic graphical object
  geom_line(aes(y=result$actual.tr5), colour="green") +  # first layer
  geom_line(aes(y=result$prediction.5), colour="red")  # second layer

RMSE = c(((mean((result$actual.tr-result$prediction.1)^2))^0.5)
         ,((mean((result$actual.tr2-result$prediction.2)^2))^0.5)
         ,((mean((result$actual.tr3-result$prediction.3)^2))^0.5)
         ,((mean((result$actual.tr4-result$prediction.4)^2))^0.5)
         ,((mean((result$actual.tr5-result$prediction.5)^2))^0.5)
)