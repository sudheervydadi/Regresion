# 1. Basics (Clearing the environment and set the working directory)====

# Remove all the objects in the environment
rm(list = ls())

# Set the working directory

# 2. Read the data and do the exploratory data analysis ====
data<-read.csv("CustomerData.csv",header=T)

dim(data)
names(data)
str(data)
summary(data)

# 3. Data pre-processing ====

# 3a. Train - Test split ====
library(caret)
set.seed(123)
sample_rows = createDataPartition(data$TotalRevenueGenerated,p=0.7,list = F)
train = data[sample_rows,]
validation = data[-sample_rows,]

# 3b. Dropping some columns ====
train$CustomerID <- NULL
validation$CustomerID <- NULL

# 3c. Data type conversion ====
train$City <- as.factor(train$City)
validation$City <- as.factor(validation$City)

# 3d. Imputing missing values ====
sum(is.na(train))
sum(is.na(validation))

# 4. Model1 - Basic model ====

# 4a. Model building and summary
model1 <- lm(formula = TotalRevenueGenerated ~ ., data = train)
summary(model1)

# 4b. Residual analysis
par(mfrow=c(2,2))
plot(model1)

# 4c. Model metrics
model1_train_preds <- model1$fitted.values #OR
model1_train_preds <- predict(object = model1, newdata = train)
model1_validation_preds <- predict(object = model1,
                                   newdata = validation)


library(DMwR)
m1_train = data.frame(m1_train = regr.eval(trues = train$TotalRevenueGenerated, 
          preds = model1_train_preds))
m1_test = data.frame(m1_test = regr.eval(trues = validation$TotalRevenueGenerated, 
          preds = model1_validation_preds))
performance_df = cbind(m1_train,m1_test)

# 5. Model2 - Removing outliers and influential points ====

# 5a. Finding and removing outliers
lev = hat(model.matrix(model1)) #gives all leverages
plot(lev)
max_lev = which(lev>0.2)


#Leverage
#Finding the threshold and removing them from train data
train[lev>0.2,]
nrow(train[lev>0.2,])


#cooks distance
cook = cooks.distance(model1)
plot(cook,ylab="Cooks distances")

max_cook=as.numeric(which.max(cook))
points(max_cook,cook[max_cook],col='red', pch=19)

remove_points = unique(c(max_lev,max_cook))
train <- train[-remove_points,]

#Residual outliers
residuals = model1$residuals
outliers <- boxplot(residuals,plot=T)$out
sort(outliers)
length(outliers)

# 5b. Model building and summary
model2<- lm(TotalRevenueGenerated~., data = train)
summary(model2)

# 5c. Model metrics
model2_train_preds <- predict(object = model2, newdata = train)
model2_validation_preds <- predict(object = model2, newdata = validation)

m2_train = data.frame(m2_train = regr.eval(trues = train$TotalRevenueGenerated, preds = model2_train_preds))
m2_test = data.frame(m2_test = regr.eval(trues = validation$TotalRevenueGenerated, preds = model2_validation_preds))
performance_df = cbind(performance_df,m2_train,m2_test)

# 6. Model3 - Handling Multicollinearity using VIF ====

install.packages('car')

# 6a. Identifying Multicollinear variables using VIF
library(car)
vif_m2 = vif(model2)
keep_cols = names(vif_m2[vif_m2<2])
keep_cols = c(keep_cols,'TotalRevenueGenerated')
train_vif = train[keep_cols]
validation_vif = validation[keep_cols]

# 6b. Model building and summary
model3 <- lm(TotalRevenueGenerated ~., data=train_vif)
summary(model3)

# 6c. Model metrics
model3_train_preds <- predict(object = model3, newdata = train_vif)
model3_validation_preds <- predict(object = model3, newdata = validation_vif)

m3_train = data.frame(m3_train = regr.eval(trues = train_vif$TotalRevenueGenerated, preds = model3_train_preds))
m3_test = data.frame(m3_test  =regr.eval(trues = validation_vif$TotalRevenueGenerated, preds = model3_validation_preds))
performance_df = cbind(performance_df,m3_train,m3_test)

# 7. Model4 - Handling Multicollinearity using Step-AIC ====

# 7a. Identifying required variables using Stepwise regression
library(MASS)
model1 <- lm(formula = TotalRevenueGenerated ~ ., data = train)
summary(model1)
step <- stepAIC(model1, direction="both")
step$call

# 7b. Model building and summary
model4 = lm(formula = TotalRevenueGenerated ~ City + NoOfChildren + MinAgeOfChild + 
              MaxAgeOfChild + Tenure + FrquncyOfPurchase + NoOfUnitsPurchased + 
              FrequencyOFPlay + NoOfGamesPlayed + NoOfGamesBought + FavoriteChannelOfTransaction + 
              FavoriteGame, data = train)
summary(model4)

# 7c. Model metrics
model4_train_preds <- predict(object = model4, newdata = train)
model4_validation_preds <- predict(object = model4, newdata = validation)

m4_train = data.frame(m4_train = regr.eval(trues = train$TotalRevenueGenerated, preds = model4_train_preds))
m4_test = data.frame(m4_test= regr.eval(trues = validation$TotalRevenueGenerated, preds = model4_validation_preds))
performance_df = cbind(performance_df,m4_train,m4_test)


