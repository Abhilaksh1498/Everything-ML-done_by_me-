dataset = read.csv('Mall_Customers.csv')
X = dataset[,4:5]

X = scale(X)

dendogram = hclust(dist(X, method = 'euclidean'),method = 'ward.D')
plot(dendogram,
     main = paste('Dendogram'),
     xlab = 'Customers',
     ylab = 'Distance')

hc= dendogram
y_hc= cutree(dendogram, k=5)

clusplot(X, 
         y_hc,
         span = TRUE,
         color = TRUE,
         shade = TRUE,
         lines = 0,
         xlab = 'Annual Income',
         ylab = 'Expenditure',
         main = paste('Clusters of Customers'))
