#Data Preprocessing ; importing the

dataset = read.csv('Salary_Data.csv')
#dataset = dataset[, 2:3]

#Splitting the dataset in to training set and test set
#install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(dataset$Salary, SplitRatio = 2/3 )
training_set = subset(dataset, split == TRUE)
test_set= subset(dataset, split == FALSE)

#Feature scaling
# training_set[, 2:3] = scale(training_set[, 2:3])
# test_set[, 2:3] = scale(test_set[, 2:3])

#Fitting simple linear regression to the Training set
regressor = lm(formula = Salary ~ YearsExperience,
               data = training_set)
#summary(regressor) - in console to see the relavent info about regressor

#predicting the test set results
y_pred = predict(regressor, test_set)

#Visualising the Training set results
#install.packages('ggplot2')
library(ggplot2)
ggplot() +
  geom_point(aes(x = training_set$YearsExperience , y = training_set$Salary),
             color = 'red') +
  geom_line(aes(x = training_set$YearsExperience, y = predict(regressor,
                                                              newdata = training_set)),
                color = 'blue') +
  ggtitle("Salary Vs. Experience (Training set)") +
  xlab("YearsExperiece") +
  ylab("Salary")

#Visualising the Test set results
ggplot() +
  geom_point(aes(x = test_set$YearsExperience , y = test_set$Salary),
             color = 'red') +
  geom_line(aes(x = training_set$YearsExperience, y = predict(regressor,
                                                              newdata = training_set)),
            color = 'blue') +
  ggtitle("Salary Vs. Experience (Training set)") +
  xlab("YearsExperiece") +
  ylab("Salary")



