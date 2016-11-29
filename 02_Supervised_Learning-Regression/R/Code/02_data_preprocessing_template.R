#Data Preprocessing ; importing the

dataset = read.csv('Data.csv')
#dataset = dataset[, 2:3]

#Splitting the dataset in to training set and test set
#install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(dataset$DependentVariable, SplitRatio = 0.8 )
training_set = subset(dataset, split == TRUE)
test_set= subset(dataset, split == FALSE)

#Feature scaling
# training_set[, 2:3] = scale(training_set[, 2:3])
# test_set[, 2:3] = scale(test_set[, 2:3])

