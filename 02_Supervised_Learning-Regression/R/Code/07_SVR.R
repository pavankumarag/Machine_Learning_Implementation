#SVR 
#Regression template


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

#Fitting the polynomial SVR regression model to the dataset
#create a regressor
#install.packages('e1071')
library(e1071)
regressor = svm( Salary ~ .,
                 data = dataset,
                 type = 'eps-regression')
#Predict a new result with SVR
y_pred = predict(regressor, newdata = data.frame(Level = 6.5))

#Visualising the  SVR regression model results
library(ggplot2)
ggplot() +
  geom_point(aes(x = dataset$Level, y = dataset$Salary),
             color = 'red') +
  geom_line(aes(x = dataset$Level, y = predict(regressor, newdata = dataset)),
            color = 'blue') +
  ggtitle("Truth or Bluff(Regression Model)") +
  xlab("Level") +
  ylab("Salary")

#Visualising the  SVR regression model results
#For high resolution and smoothen the curve
library(ggplot2)
X_grid = seq(min(dataset$Level), max(dataset$Level), 0.1)
ggplot() +
  geom_point(aes(x = dataset$Level, y = dataset$Salary),
             color = 'red') +
  geom_line(aes(x = X_grid, y = predict(regressor, newdata = data.frame(Level = X_grid))),
            color = 'blue') +
  ggtitle("Truth or Bluff( SVR Regression Model)") +
  xlab("Level") +
  ylab("Salary")
