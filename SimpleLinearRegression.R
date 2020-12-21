#####The purpose of this activity is to create a simple linear regression model and observe the residuals - during this course,
# we'll have to perform a bit of EDA and data preprocessing as well

# Clear the environment
rm(list=ls(all=TRUE))

# Set the working directory
#setwd('~/Desktop/')
# Read the data file

toyota_data <- read.csv('Toyota_SimpleReg.csv',header = TRUE,sep = ',')

##### Data Exploration

# Check the first few observations of the data
head(toyota_data)

# Print shape of the data
cat('Shape of the data is:',dim(toyota_data))

# Function to print number of rows, number of columns, and column names
data_summary <- function(df){
  return(list(nrow(df),ncol(df),colnames(df)))
}

toyota_summary <- data_summary(toyota_data)
toyota_summary[[3]]

# Check the structure of data
str(toyota_data)
# 'data.frame':	1436 obs. of  4 variables:
# $ Id       : int  1 2 3 4 5 6 7 8 9 10 ...
# $ Model    : Factor w/ 372 levels "\xa0TOYOTA Corolla 1.3 16V HATCHB G6 2/3-Doors",..: 332 332 67 332 331 331 64 326 62 59 ...
# $ Price    : int  13500 13750 13950 14950 13750 12950 16900 18600 21500 12950 ...
# $ Age_06_15: int  57 57 58 60 64 66 61 64 61 57 ...

# Check summary of the data
summary(toyota_data)

# Rename 'Age_06_15' to 'Age'
names(toyota_data)[names(toyota_data) %in% 'Age_06_15'] <- 'Age'

# Check for missing values
sum(is.na(toyota_data))

# There are no missing values in the dataset - there is no need to do imputation

# Find column wise missing values
apply(toyota_data,2,function(x) sum(is.na(x)))
# or
colSums(is.na(toyota_data))

# Create a scatter plot and correlation plot between Price and Age of the car to form a hypothesis
plot(x = toyota_data$Age,y = toyota_data$Price,main = 'Price vs Age',xlab = 'Age (in yrs)',ylab = "Price (in '000$)",
     col='blue')
grid(10,10,lwd = 1,col='Black')

library(corrplot)
corrplot(cor(toyota_data[c('Price','Age')]),method = 'number',title = 'Correlation Matrix between Price and Age of the cars')
# We can observe by looking at both the correlation matrix and the scatter plot there is an inverse relationship between Price and Age

#####Data Preprocessing

# Drop the columns 'Id' and 'Model' for our purpose - otherwise, we could have used 'Model' column to engineer new features as 
# a car's price depends on the model type and number of doors as well
vars_to_drop <- c('Id','Model')
toyota_data[vars_to_drop] <- NULL
# or toyota_data <- subset(toyota_data,select=c('Age','Price'))

#We can standardize the dataset, but as far as simple linear regression result is concerned, that won't have any effect

# Split the dataset into train and test sets
set.seed(234)
library(caTools)
sample_idx <- sample.split(toyota_data$Price,SplitRatio = 0.7)
train <- toyota_data[sample_idx==TRUE,]
test <- toyota_data[sample_idx==FALSE,]

# A quick check on the distribution of y-variable after splitting the data
par(mfrow=c(1,2))
hist(train$Price,main = 'Histogram for Price(train)',xlab = 'Price(train)')
hist(test$Price,main = 'Histogram for Price(test)',xlab = 'Price(test)')

#####Create a Simple Linear Regression model between Price (target) and Age (feature)
LinReg <- lm(formula = Price~Age,data = toyota_data)

summary(LinReg)

#####Residual Plots
par(mfrow=c(2,2))
plot(LinReg)
# a) The relationship seems to be linear. In real life dataset, the red line will never be exactly zero.
# b) There seems to be a bit of heteroskedasticity - that is a subjective call however. You can adopt various ways, such as doing
# transformation or creating different models on different subsets of data
# c) The errors seem to be distributed more or less normally - however, there are a few points that do not lie on theoretecal 
# Q-Q line. Again, models on real-life datasets will exhibit these patterns - it is better you perform some statistical test 
# for normality.
# d) The last plot shows cook's distance, based on which you can inspect outliers

####Function Equation (or linear regression equation)
# Price = 26105.801 - 170.934*Age - a unit increase in car age reduces the car price by 170 units - you can plot this relationship as:
par(mfrow=c(1,1))
plot(toyota_data$Age,toyota_data$Price,xlab="Age of the Car",ylab="Price in ($)",
     main="Car Price Vs. Age: Best fit line", col= "blue")
abline(LinReg,col="red",lwd=1)

####Prediction on train and test datasets and model evaluation
train_preds <- predict(LinReg,train)
test_preds <- predict(LinReg,test)

options(scipen=10)
library(DMwR)
# DMwR has certain inbuilt error metrics - you can however free to define the metrics for your particular use case and then evaluate
regr.eval(trues = train$Price,preds = train_preds)
regr.eval(trues = test$Price,preds = test_preds)
