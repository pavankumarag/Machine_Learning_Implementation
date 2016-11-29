#Data Preprocessing ; importing the

dataset = read.csv('50_Startups.csv')
#dataset = dataset[, 2:3]

#Encoding categorical data
dataset$State = factor(dataset$State,
                         levels = c('New York', 'California', 'Florida'),
                         labels = c(1,2,3))

#Splitting the dataset in to training set and test set
#install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(dataset$Profit, SplitRatio = 0.8 )
training_set = subset(dataset, split == TRUE)
test_set= subset(dataset, split == FALSE)

#Feature scaling
# training_set[, 2:3] = scale(training_set[, 2:3])
# test_set[, 2:3] = scale(test_set[, 2:3])

#Fitting Multiple Linear Regression to the Training set
#regressor = lm(formula = Profit ~ R.D.Spend + Administration + Marketing.Spend + State) # or we can express as below
regressor = lm(formula =  Profit ~ .,
               data = training_set)
summary(regressor)
#Based on the p-value as seen from summary, only R.D.Spend is the most statistically significant variable to predict profit hence
#regressor = lm(formula = Profit ~ R.D.Spend,
#               data = training_set)
y_pred = predict(regressor, newdata = test_set)

#Building the optimal model using Backward Elimination
regressor = lm(formula = Profit ~ R.D.Spend + Administration + Marketing.Spend + State,
               data = training_set)
#or we can use data = dataset in the above line
summary(regressor)
regressor = lm(formula = Profit ~ R.D.Spend + Administration + Marketing.Spend,
               data = training_set)
# Optional Step: Remove State2 only (as opposed to removing State directly)
# regressor = lm(formula = Profit ~ R.D.Spend + Administration + Marketing.Spend + factor(State, exclude = 2),
#                data = dataset)
summary(regressor)
regressor = lm(formula = Profit ~ R.D.Spend  + Marketing.Spend,
               data = training_set)
summary(regressor)
regressor = lm(formula = Profit ~ R.D.Spend ,
               data = training_set)
summary(regressor)




