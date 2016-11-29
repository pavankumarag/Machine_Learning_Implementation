#Polynomial regression

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

#Fitting the linear regression model to the dataset
lin_reg = lm(formula = Salary ~ .,
             data = dataset)

#Fitting the polynomial regression model to the dataset
dataset$Level2 = dataset$Level^2
dataset$Level3 = dataset$Level^3
#addding new polynomial degree to fit curve closer
dataset$Level4 = dataset$Level^4
poly_reg = lm(formula = Salary ~ .,
              data = dataset)


#Visualising the linear regression results
#install.packages('ggplot2')
library(ggplot2)
ggplot() +
  geom_point(aes(x = dataset$Level, y = dataset$Salary),
             color = 'red') +
  geom_line(aes(x = dataset$Level, y = predict(lin_reg, newdata = dataset)),
            color = 'blue') +
  ggtitle("Truth or Bluff(Linear Regression)") +
  xlab("Level") +
  ylab("Salary")

#Visualising the polynomial regression results
ggplot() +
  geom_point(aes(x = dataset$Level, y = dataset$Salary),
             color = 'red') +
  geom_line(aes(x = dataset$Level, y = predict(poly_reg, newdata = dataset)),
            color = 'blue') +
  ggtitle("Truth or Bluff(Polynomail Regression)") +
  xlab("Level") +
  ylab("Salary")

#Predict a new result with linear regression 
y_pred = predict(lin_reg, newdata = data.frame(Level = 6.5))

#Predict a new result with polynomial regression
y_pred_poly = predict(poly_reg, newdata = data.frame(Level = 6.5,
                                                Level2 = 6.5^2,
                                                Level3 = 6.5^3,
                                                Level4 = 6.5^4))
