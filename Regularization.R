#2. Regularization ====

# Read data, Imputation, train test split ====
rm(list = ls())
auto_data = read.table("auto-data")

head(auto_data)
# Specifiying column headers
names(auto_data) <- c("mpg","cylinders","displacement",
                      "horsepower","weight","acceleration",
                      "model year","origin","car name")


# Removing data points with mpg as NA
auto_data = auto_data[!is.na(auto_data$mpg),]

# Removing column 'car name'
auto_data$`car name` <- NULL

# Converting categorical attributes to factors
cat_attr <- c('origin','cylinders')
auto_data[,cat_attr] <- data.frame(lapply(auto_data[,cat_attr], factor))

# Checking column-wise NA values
sapply(auto_data,function (x) sum(is.na(x)))

# NA imputation
#library(DMwR)
#imputed_data <- centralImputation(auto_data)


# Train-Validation data split
library(caret)
set.seed(654)
train_rows <- createDataPartition(auto_data$mpg,p=0.7,list=FALSE)
train_data <- auto_data[train_rows,]
val_data <- auto_data[-train_rows,]

sapply(train_data,function (x) sum(is.na(x)))
sapply(val_data,function (x) sum(is.na(x)))

library(DMwR)
train_data<-knnImputation(train_data)
val_data<-knnImputation(val_data,distData = train_data)


# Decoupling Target to avoid accidental mistakes
train_target <- train_data$mpg
val_target <- val_data$mpg

train_data$mpg <- NULL
val_data$mpg <- NULL

## Regualized apporach  ====

train_final = train_data
val_final = val_data

## Dummification
library(dummies)

# Creating dummy object
dummy_obj <- dummyVars(~ cylinders + origin , data = train_final)
# Using predict function to dummify train data
train_dummies <- predict(dummy_obj,newdata=train_final)
# Deleting categorical columns from train data
train_final[,c('cylinders','origin')] <- NULL
# Appending dummified columns to train data
train_final <- cbind(train_final,train_dummies)

# Using predict function to dummify validation data
val_dummies <- predict(dummy_obj,newdata=val_data)
# Deleting categorical columns from validation data
val_final[,c('cylinders','origin')] <- NULL
# Appending dummified columns to validation data
val_final <- cbind(val_final,val_dummies)


## Ridge and Lasso Regresison ====
library(glmnet)

# Ridge regression
ridge_model <- glmnet(as.matrix(train_final), train_target, alpha = 0, family = 'gaussian')
# Plotting co-efficients vs lambda values
plot(ridge_model, xvar = 'lambda', main = 'Ridge coefficients vs log(lambda)')

# Lasso regression
lasso_model <- glmnet(as.matrix(train_final), train_target, alpha = 1, family = 'gaussian')
# Plotting co-efficients vs lambda values
plot(lasso_model, xvar = 'lambda', main = 'Lasso coefficients vs log(lambda)')


# Dataframe for error analysis ====
regularization_analysis <- df <- data.frame(matrix(ncol = 3))
names(regularization_analysis)= c('Alpha','Train_RMSE','Validation_RMSE')

## Iterative process with Elastic-Net to find out best alpha value (balance between Ridge and Lasso) ====
for (a in seq(0,1,0.1))
{
  # Elastic net model
  elastic_net <- cv.glmnet(as.matrix(train_final), train_target, alpha = a, family = 'gaussian')
  
  # Train-Validation predictions
  reg_train_preds <- predict(elastic_net, as.matrix(train_final), s=elastic_net$lambda.1se)
  reg_val_preds <- predict(elastic_net, as.matrix(val_final), s=elastic_net$lambda.1se)
  
  # RMSE calculation
  reg_train_rmse <- RMSE(pred = reg_train_preds, obs = train_target)
  reg_test_rmse <- RMSE(pred = reg_val_preds, obs = val_target)
  
  # Appending entry to dataframe for later analysis
  regularization_analysis <- rbind(regularization_analysis, c(a,reg_train_rmse,reg_test_rmse))
}

# Removing 1st NA row
regularization_analysis <- na.omit(regularization_analysis)

ggplot(regularization_analysis, aes(x = Alpha))+ 
  geom_point(aes(x = Validation_RMSE, y = Train_RMSE))+ 
  ylab('Train_RMSE') + xlab('Validation_RMSE') + geom_text(aes(x = Validation_RMSE, y = Train_RMSE,label=Alpha))

# Final model selection ====
## Validation error seems better at alpha = 0.6
## You can select either one of these as your final model
## Final Model
final_model <- cv.glmnet(as.matrix(train_final), train_target, alpha = 0.6,family = 'gaussian')

## Final Predictions
final_train_preds <- predict(final_model, as.matrix(train_final), s=final_model$lambda.1se)
final_val_preds <- predict(final_model, as.matrix(val_final), s=final_model$lambda.1se)

reg_train_rmse <- RMSE(pred = final_train_preds, obs = train_target)
reg_test_rmse <- RMSE(pred = final_val_preds, obs = val_target)
reg_train_rmse;reg_test_rmse
