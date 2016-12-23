#Hierarchical Clustering
dataset = read.csv('Mall_Customers.csv')
X = dataset[4:5]

#Using the dendogram to find the optimal number of clusters
dendogram = hclust(dist(X, method='euclidean'), method='ward.D')
plot(dendogram, main = paste("Dendogram"), xlab='customers',ylab ='Euclidean distances')

#Fitting hierarchial clustering to the mall dataset
hc = hclust(dist(X, method='euclidean'), method='ward.D')
y_hc = cutree(hc, 5)

#Visualsing the clusters
library(cluster)
clusplot(X, y_hc, lines=0, shade=TRUE, color=TRUE, labels=2,plotchar=FALSE,
         span=TRUE, main=paste("Cluster of clients"), xlab="Annual income",
         ylab="Spending score")
