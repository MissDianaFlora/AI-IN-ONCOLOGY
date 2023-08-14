#Import Data
library(readxl)
Data <- read_excel("~/BMSF 1/YEAR 3/analysis 2023/Multiple Myeloma Data/ML Dataset.xlsx")
View(Data)
#Machine Learning Models
#1. Data preprocessing
#checking missing data
summary(is.na(Data$Ecog_index))
summary(is.na(Data$`1st Regimen`))
summary(is.na(Data$age))
summary(is.na(Data$Sex))
summary(is.na(Data$ISS_stage))
summary(is.na(Data$renalfailure))
summary(is.na(Data$anemia))
summary(is.na(Data$`Bone Pain`))
summary(is.na(Data$`remission status`))


#check the data type
is.numeric(Data$age)
is.numeric(Data$`Survival Time (Months)`)
is.factor(Data$Ecog_index)
is.factor(Data$renalfailure)
is.factor(Data$anemia)
is.factor(Data$`Bone Pain`)
is.factor(Data$`remission status`)
is.factor(Data$ISS_stage)
is.factor(Data$Sex)
is.factor(Data$`1st Regimen`)
is.factor(Data$y)

#convert to factor
Data$Ecog_index = as.factor(Data$Ecog_index)
Data$renalfailure = as.factor(Data$renalfailure)
Data$anemia = as.factor(Data$anemia)
Data$`Bone Pain` = as.factor(Data$`Bone Pain`)
Data$`remission status` = as.factor(Data$`remission status`)
Data$ISS_stage = as.factor(Data$ISS_stage)
Data$Sex = as.factor(Data$Sex)
Data$`1st Regimen` = as.factor(Data$`1st Regimen`)
Data$y = as.factor(Data$y)

#convert to numeric
Data$age = as.numeric(Data$age)
Data$`Survival Time (Months)`= as.numeric(Data$`Survival Time (Months)`)
Data$`Mcomponent_g/l` = as.numeric(Data$`Mcomponent_g/l`)


#checking for duplicates
duplicated(Data$Hosp_ID)

#KEEP only variable that you want to use (age, sex,ecog, regimen, remission status, bone pain, renal failure, hypercalcemia, anemia, survval status, survival time)
dataset = Data[c(2,3,4,5,6,7,8,9,10,11,12,13,14)]


#Building a training set and test set
library(caTools)
set.seed(123)
split=sample.split(dataset$y, SplitRatio = 0.75)
MMtraining_set=subset(dataset, split==TRUE)
MMtest_set=subset(dataset, split==FALSE)

#Feature scaling
#use either standardization or normalization
# Convert non-numeric columns to numeric (if needed)
MMtraining_set[, c(2,6,10)] <- sapply(MMtraining_set[, c(2,6,10)], as.numeric)
MMtest_set[, c(2,6,10)] <- sapply(MMtest_set[, c(2,6,10)], as.numeric)


# Now, scale the selected columns
MMtraining_set[, c(2,6,10)] <- scale(MMtraining_set[, c(2,6,10)])
MMtest_set[, c(2,6,10)] = scale(MMtest_set[, c(2,6,10)])


#Fitting the Logistic Regression model on the training set
#Age, spep, ecog, survival time, 1st regimen, remission status, anemia, sex, ISS stage, renal failure, hypercalcemia, bone pain
# Fit logistic regression using Firth's penalized likelihood
logisticclassifier = glm(formula = y ~ ., family = binomial,data = MMtraining_set)
summary(logisticclassifier)
#predicting the test set results using test data
logistic_pred = predict(logisticclassifier, type = 'response', newdata = MMtest_set[-1])

#converting the predicted values into 0 and 1
logisticpred = ifelse(logistic_pred > 0.5, 1, 0)


#Convert y to a factor or vector using unlist()
y <- as.factor(unlist(MMtest_set[, 1]))

#evaluating the predicted values using the confusion matrix
#The Confusion Matrix, it counts the number of correct predictions and number of incorrect predictions
cm = table(y, logisticpred)

#the number of correct predictions is 2
#the number of incorrect predictions is 3


#Decision Tree in Classification
library(rpart)

#fitting the decision tree classifier
decisiontreeclassifier = rpart(formula = y ~ ., data = MMtraining_set)
summary(decisiontreeclassifier)


#predict the test dataset using decision tree
decisionpred = predict(decisiontreeclassifier, newdata = MMtest_set[-1], type = 'class')



#Convert y to a factor or vector using unlist()
y <- as.factor(unlist(MMtest_set[, 1]))

#evaluating the predicted values using the confusion matrix
#The Confusion Matrix, it counts the number of correct predictions and number of incorrect predictions
cm = table(y, decisionpred)

#plot tree
install.packages("party")
library(party)
outputtree = ctree(y ~ age + `Survival Time (Months)` + `Mcomponent_g/l`,data = MMtraining_set)
plot(outputtree)



#Random Forest classification
install.packages("randomForest")
library(randomForest)

#create the random forest predictor using training set 
randomforestclassifier = randomForest(x = MMtraining_set[-1], y = MMtraining_set$y, ntree = 10)
summary(randomforestclassifier)


#predict the test dataset using decision tree
random_pred = predict(randomforestclassifier, newdata = MMtest_set[-1])
random_pred = ifelse(prob_pred > 0.5, 1, 0)

#Convert y to a factor or vector using unlist()
y <- as.factor(unlist(MMtest_set[, 1]))

#evaluating the predicted values using the confusion matrix
#The Confusion Matrix, it counts the number of correct predictions and number of incorrect predictions
cm = table(y, random_pred)


#xgBoost 
#Building a training set and test set because feature scaling is not required here
dataset1 = Data[c(2,3,7,11)]
library(caTools)
set.seed(123)
split=sample.split(dataset1$y, SplitRatio = 0.75)
training_set=subset(dataset1, split==TRUE)
test_set=subset(dataset1, split==FALSE)


# Fitting XG Boost to the training set
install.packages("xgboost")
library(xgboost)


#implement XGBoost
library(xgboost)

# Convert the training data frame to a matrix
xgboostclassifier = xgboost(data = as.matrix(training_set[-1]), label = training_set$y, nrounds = 10)





#survival analysis
library(survival)
library(survminer)
install.packages("survex")
install.packages("randomForestSRC")
library(survex)
library(randomForestSRC)


library(caTools)
set.seed(123)
split=sample.split(dataset$y, SplitRatio = 0.75)
training_set=subset(dataset, split==TRUE)
test_set=subset(dataset, split==FALSE)


surv_obj <- Surv(training_set$`Survival Time (Months)`, training_set$y)
# Fit the survival model on the training set
cox_model <- coxph(surv_obj ~ ., data = training_set)

# Evaluate the model on the test set
predictions <- predict(cox_model, newdata = test_data, type = "expected")
# Perform evaluation using appropriate metrics for survival analysis

surv_obj <- Surv(`Survival Time (Months)`, y)

# Now, you can use the 'surv_obj' in your survival analysis models
# For example, fitting a Cox proportional hazards model
cox_model <- coxph(surv_obj ~ predictor1 + predictor2 + ..., data = your_data)













