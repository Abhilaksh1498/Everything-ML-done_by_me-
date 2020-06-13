dataset = read.csv('Mall_Customers.csv')
X= dataset[,4:5]


wcss= vector()
for (i in 1:10) wcss[i]= sum(kmeans(X,i)$withinss) 

plot(x=1:10, y=wcss, type = 'b')

set.seed(29)
kmeans = kmeans(X,6)
clusplot(X, 
         kmeans$cluster,
         span = TRUE,
         color = TRUE,
         shade = TRUE,
         lines = 0,
         xlab = 'Annual Income',
         ylab = 'Expenditure',
         main = paste('Clusters of Customers'))