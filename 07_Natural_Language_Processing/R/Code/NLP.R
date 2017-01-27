#Natural Language Processing

dataset_org = read.delim("Restaurant_Reviews.tsv", quote = '', stringsAsFactors = FALSE)

#Cleaning the texts
#install.packages('tm')
#install.packages('SnowballC')
library(tm)
library(SnowballC)
corpus = VCorpus(VectorSource(dataset$Review))
#as.character(corpus[[1]]) #displays the first review
corpus = tm_map(corpus, content_transformer(tolower))
corpus = tm_map(corpus, removeNumbers)  # check here ; as.character(corpus[[841]])
corpus = tm_map(corpus, removePunctuation) # check here ; as.character(corpus[[1]])
corpus = tm_map(corpus, removeWords, stopwords())  # check here ; as.character(corpus[[1]])
corpus = tm_map(corpus, stemDocument) # check here ; as.character(corpus[[1]])
corpus = tm_map(corpus, stripWhitespace) # check here ; as.character(corpus[[841]])

#Creating the Bags of Words model
dtm = DocumentTermMatrix(corpus)
dtm = removeSparseTerms(dtm, 0.999)

#Build the Machine learning classification model(Random Forest Classification)
dataset = as.data.frame(as.matrix(dtm))
dataset$Liked = dataset_org$Liked

#Encoding the target feature as factor
dataset$Liked = factor(dataset$Liked, levels = c(0,1))

#Splitting the dataset in to training set and test set
#install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(dataset$Liked, SplitRatio = 0.8 )
training_set = subset(dataset, split == TRUE)
test_set= subset(dataset, split == FALSE)


#Fitting Random Forest classifier in to the Training set
#Create your Classifier here
library(randomForest)
set.seed(123)
classifier = randomForest(x = training_set[-692], y = training_set$Liked,
                          ntree = 10)

#Predicting the test set
y_pred = predict(classifier, newdata = test_set[-692])

#Making the confusion matrix
cm = table(test_set[, 692], y_pred)

#Accuracy = (78+72) / 200