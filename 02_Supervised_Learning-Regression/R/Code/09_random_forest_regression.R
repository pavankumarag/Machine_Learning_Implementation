#Random Forest regression 

#Data Preprocessing ; importing the

dataset = read.csv('Position_Salaries.csv')
dataset = dataset[, 2:3]

#Splitting the dataset in to training set and test set
#install.packages('caTools')
# library(caTools)
# set.seed(123)
# split = sample.split(dataset$Salary, SplitRatio = 0.8 )
# training_set = subset(dataset, split == TRUE)
# test_set= subset(dataset, split == FALSE)

#Feature scaling
# training_set[, 2:3] = scale(training_set[, 2:3])
# test_set[, 2:3] = scale(test_set[, 2:3])

#Fitting the Random Forest regression model to the dataset
#create a regressor
#install.packages('randomForest')
library('randomForest')
set.seed(1234)
#regressor = randomForest(x = dataset[1], y = dataset$Salary, ntree = 10)
#regressor = randomForest(x = dataset[1], y = dataset$Salary, ntree = 100)
regressor = randomForest(x = dataset[1], y = dataset$Salary, ntree = 300)

#Predict a new result with
y_pred = predict(regressor, newdata = data.frame(Level = 6.5))

#Visualising the  Random Forest regression model results
#Since this regression is non continous, we use high resolution
# library(ggplot2)
# ggplot() +
#   geom_point(aes(x = dataset$Level, y = dataset$Salary),
#              color = 'red') +
#   geom_line(aes(x = dataset$Level, y = predict(regressor, newdata = dataset)),
#             color = 'blue') +
#   ggtitle("Truth or Bluff(Regression Model)") +
#   xlab("Level") +
#   ylab("Salary")

#Visualising the Random Forest regression model results
#For high resolution and smoothen the curve
library(ggplot2)
#X_grid = seq(min(dataset$Level), max(dataset$Level), 0.1)
X_grid = seq(min(dataset$Level), max(dataset$Level), 0.01)
ggplot() +
  geom_point(aes(x = dataset$Level, y = dataset$Salary),
             color = 'red') +
  geom_line(aes(x = X_grid, y = predict(regressor, newdata = data.frame(Level = X_grid))),
            color = 'blue') +
  ggtitle("Truth or Bluff(Random Forest Regression Model)") +
  xlab("Level") +
  ylab("Salary")
