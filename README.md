![alt text](https://github.com/ggeop/Flag-Study/blob/master/imgs/cover.PNG)
# Flag Study Analysis
## Introduction
Religion in national symbols can often be found in national anthems of flags. In this project we will try to classify the national flags according to their characteristics. It’s commonly known that a national flag is designed with specific meanings for its colors and symbols. The colors of a national flag may be worn from the people of a nation to show their patriotism, or the design of a national flag may be altered after the occurrence of important historical events. In the first part of the project we will try to predict the religion of a country according to their flag characteristics. Although, in the second part, we will try to create groups of flags that have common characteristics, clusters.

## Dataset
The dataset contains various details of countries and their flags (in this dataset there are flags of countries that are not still exist). In more details, the dataset contains 194 countries and 30 variables-attributes for each country. We have split the dataset into tree tables according to the type of the variable. In the Table 1 - Geographical Characteristics we have variables that are related to the geographical characteristics of the flag, in the Table 2 – Colour Characteristics we have the variables related to the colours of the flag and in the Table 3 - Geometrical Characteristics we have all the geometrical characteristics (e.g. shapes, lines).

![alt text](https://github.com/ggeop/Flag-Study/blob/master/tables/table_1.PNG)

![alt text](https://github.com/ggeop/Flag-Study/blob/master/tables/table_2.PNG)

![alt text](https://github.com/ggeop/Flag-Study/blob/master/tables/table_3.PNG)

## PART I – Classification
In this part of the project we will try to categorize the nation flags and identify their religion according to the characteristics of the flag. In the beginning of this analysis we do a descriptive analysis in order to understand better our data, something very important if we want to understand the behavior of the following models. We will try different classification algorithms (classifiers) and finally we compare our results and choose one of them. In more details, our classification analysis will include the following methods:

* Decision Trees
* upport Vector Machines

### Descriptive Analysis
First, we import the data in R studio. At this stage we will take a quick view in the data in order to understand them better (Table 2 – Colour Characteristics and Table 3 - Geometrical Characteristics). 
To begin, we create a bar graph with the number of the flags for each country (see Figure 1 - Number of flags for each Religion). Christians and Catholic are the most widespread religions.

![alt text](https://github.com/ggeop/Flag-Study/blob/master/figures/figure_1.PNG)

Respectively, we create bar plots for colour characteristics (see in Figure 3 –Number of Colours and Figure 4 - Main Hue Colour). We observe that the majority of flags are composed with 3 different colours (see in the left plot) and the most popular colour is Red.

![alt text](https://github.com/ggeop/Flag-Study/blob/master/figures/figure_3.PNG)

![alt text](https://github.com/ggeop/Flag-Study/blob/master/figures/figure_4.PNG)

### Classification Trees
In this part of the project we will create various classification trees (recursive partitioning). Firstly, we will create a maximal tree and after we will train other two different simpler models by using different methods. In more details, after the initial tree we will try to reduce the variables and grow again a new tree. Finally, we will use another approach by using a more pioneer method by using Rpart  library. Hence, at the end we have to combine the results of models and keep the model with highest prediction abilities.

#### 1.	CART Tree
Firstly we create our initial model, after we create other models by using different methods and finally we compare our results. For model evaluation we used an algorithm which do K-fold Cross validation. We preferred this technique because our dataset has only 194 observations, a very small number, which isn’t enough for a simple split for training and test dataset. Our major criterion for model evaluation is the model accuracy. Accuracy is the number of good predictions divided with the total predictions.
The model after tree function (see in Figure 5 – Initial Classification Tree) is very complicated because we have a lot of mixed variables (numerical and categorical).

![alt text](https://github.com/ggeop/Flag-Study/blob/master/figures/tree_1.PNG)

I would like to mention that we used “Gini” method for node splitting because we have categorical target variable, so we can’t use “variance” method. This tree is the largest tree that we could have with 30 terminal leaves. It’s very complicated; so it’s very difficult to interpret our results. We evaluate our model by using K-fold cross validation and finally to take the average of the accuracy score of each fold. The accuracy score of this model is 43%.

#### 2.	Classification Tree & Variable Selection with Random Forest
In this part we firstly do a variable selection before we train a tree model. For variable selection we used Random Forest Method. Random forest improves predictive accuracy by generating a large number of bootstrapped trees (based on random samples of variables) , classifying a case using each tree in this new "forest", and deciding a final predicted outcome by combining the results across all of the trees (an average in regression, a majority vote in classification). Breiman and Cutler's random forest approach is implemented via the Random Forest package.

With Random Forest Method we directly measure the impact of each feature on accuracy of the model. The general idea is to permute the values of each feature and measure how much the permutation decreases the accuracy of the model. Clearly, for unimportant variables, the permutation should have little to no effect on model accuracy, while permuting important variables should significantly decrease it. So, finally, we keep the 21 most important variables (see in Figure 6 - Variables Importance (Random Forest Method)).

![alt text](https://github.com/ggeop/Flag-Study/blob/master/figures/figure_6.PNG)

![alt text](https://github.com/ggeop/Flag-Study/blob/master/figures/figure_7.PNG)

After variable selection we train again the tree model (see in Figure 7 - Classification Tree (after variable selection). This tree is less complicated and has 22 terminal leaves. 

Finally, this tree has 43% accuracy, which is the same with the previous model but it’s less complicated.

#### 3.	Rpart Tree
Now, we create another tree by using Rpart library in R. The main difference from the CART Tree function is that Rpart handling of surrogate variables. Another difference is how pruning takes places. Specifically, Rpart treats differently ordinal and categorical variables. In a sense, our model from Rpart has 13 leaves (before pruning). After we prune our model in the point which minimizes the complexity parameter (cp). Finally, our pruned tree is much simpler, with 10 terminal leaves (see in Figure 8 - Pruned Classification Tree). 

![alt text](https://github.com/ggeop/Flag-Study/blob/master/figures/figure_8.PNG)

Then we evaluate this model by using K-fold cross validation and finally we take the average of the accuracy score of each fold. The accuracy score of this model is extremely higher than the previous model 95%.

### Support Vector Machines (SVM)
With Support Vector Machines (SVM), basically, we are looking for the optimal separating hyperplane between the two classes by maximizing the margin between the classes’ closest point (see in Figure 9 - Decision Tree & SVM Comparison). In R we use “e1071” package. Firstly, we tune cost and gamma model parameters with a generic function (tune.svm() ) of statistical methods using a grid search over supplied parameter ranges (I would like to mention that this method is computational intensive and it takes several minutes to finish). After we have already the two parameters, we run the model and we calculate the accuracy of this method in order to compare it with the other classification methods.

![alt text](https://github.com/ggeop/Flag-Study/blob/master/figures/figure_9.PNG)

## Models Comparison
At this stage we have to select one of the previous methods as the optimal for classification. Our selection criterion is the accuracy of the models and also the complexity of the classification tree models(see in Table 4 - Models Comparison). We calculate the accuracy for each model by using Cross Fold Validation because the volume of the data was not sufficient for a simple split to train and test partitions. As we can see, the model from Rpart library is the less complex model, with only 10 terminal leaves and the highest accuracy of all of them (95%). The support Vector Machines (SVM) method has a good accuracy value but not greater than the Rpart tree. Also, I would like to mention that the training of the SVM classifier it takes a lot of time.

![alt text](https://github.com/ggeop/Flag-Study/blob/master/tables/table_4.PNG)

## PART II – Clustering
In this part we have to do cluster analysis (clustering). In short, we have to create sets of flags in such a way that flags in the same cluster have common geometrical and colour characteristics to each other than those in the other clusters. It can be achieved by various algorithms that significantly differ in their notion of what constitutes a cluster and how to efficiently find them. We will use the following algorithms:

* PAM Clustering
* Hierarchical Clustering

Finally, we will compare our results in a sense of interpretation and clusters homogeneity.

### Clusters Distance Metric 
We need a linkage criterion in order to determine the distance between the sets of observations as a function of the pairwise distances between observations. In our dataset we have both numerical and factor variables which need different treatment. So, we prefer to use Gower distance .

The concept of Gower distance is actually quite simple. For each variable type, a particular distance metric  works well and scales to fall between 0 and 1. Then, a linear combination, using user-specified weights (most simply an average), is calculated to create the final distance matrix. The metrics used for each data type are described below:

	Quantitative (interval): range-normalized Manhattan distance
	Ordinal: variables are first ranked, then Manhattan distance is used with a special adjustment for ties
	Nominal: variables of k categories are first converted into k binary columns and then the Dice coefficient is used

Firstly, we calculate the Gower distance by using the daisy  function. If we want to check out the performance of the matrix we can do a quick search. We can search for the most similar pairs of flags, the pairs of flags which have the minimum distance; these flags are the flags of Syria and Iraq.

![alt text](https://github.com/ggeop/Flag-Study/blob/master/figures/figure_10.PNG)

Also, we can search to find the most dissimilar pairs in our Gower distance matrix. Our matrix shows that these flags are the flags of Haiti and Hong-Kong (see Figure 11 - Haiti flag (on the left), Hong Kong flag (on the right)).

![alt text](https://github.com/ggeop/Flag-Study/blob/master/figures/figure_11.PNG)

Afterwards, we have to find the optimum number of the clusters by using Silhouettes plots (see in Figure 12 - Silhouette Plot). In Silhouette plot, the better Width is the highest. In our case the best number is nine clusters.


![alt text](https://github.com/ggeop/Flag-Study/blob/master/figures/figure_12.PNG)

### Hierarchical Clustering
In this part we will use Agglomerative Hierarchical Clustering , this method is a “bottom up” approach; each observation starts in its own cluster, and pairs of clusters are merged as one moves up to the hierarchy. In order to decide which clusters should be combined, a measure of dissimilarity between sets of observations is required (a metric, in our occasion we have the Gower distance).

Firstly, we run the model and we create a dendogram with aim to have a first look to this clustering method. As we observe in Figure 13 - Hierarchical Clustering Dendrogram the 9 clusters can’t be clear identified with this method. But, we need more information if we want to have an all-around knowledge about this approach.

![alt text](https://github.com/ggeop/Flag-Study/blob/master/figures/figure_13.PNG)

In the case of that, we create Silhouette plots with aim to study the separation distance between the resulting clusters. The silhouette plots displays a measure of how close each point in one cluster is to a point in the neighboring ones and thuds it provides a way to assess parameters, like the number of clusters visually 
(see in Figure 14 -Hierarchical Clustering - Silhouette Plot). As we can see, we don’t have a very good clustering. A lot of observations look to belong to another cluster than they have assign. The red line in the plot is the limit of the lower Silhouette Width that the clusters should have. The majority of the clusters are below that line.

![alt text](https://github.com/ggeop/Flag-Study/blob/master/figures/figure_14.PNG)

### PAM Method
The PAM method is based on medoids among the observations of the dataset. These observations should represent the structure of the data. We have to find a set of nine medoids; nine clusters are constructed by assigning each observation to the nearest medoid. The goal is to find nine representative objects which minimize the sum of the dissimilarities of the observations to their closest representative object. The algorithm first looks for a good initial set of medoids (this is called the BUILD phase). Then it finds a local minimum for the objective function; such a solution there is no single switch of an observation with a medoid that will decrease the objective (this is called the SWAP phase).

We create the nine clusters by using the Pam function  in R. Firstly, we run the silhouettes plot with aim to study the separation distance between the resulting clusters (see in Figure 15 – Pam Method - Silhouette Plot). As we can see we don’t have a very well separated clusters, but we have better results than the Hierarchical Clustering method.

![alt text](https://github.com/ggeop/Flag-Study/blob/master/figures/figure_15.PNG)

### Cluster Analysis
Comparing the two clustering methods (Hierarchical Clustering and PAM Method) we observe that PAM methods create more clear clusters (based on Silhouette plots, look Figure 14 -Hierarchical Clustering - Silhouette Plot and Figure 15 – Pam Method - Silhouette Plot). So, we decide to select the PAM method. Now, we will do a descriptive analysis for each cluster in order to identify common characteristics of the clusters that maybe have sense. Firstly, we create pies with the percentage of the religions for each cluster in order to see if this attribute is important to distinguish the clusters (see in Figure 16 -Religions % per Cluster)

![alt text](https://github.com/ggeop/Flag-Study/blob/master/figures/figure_16.PNG)

As we can observe, in each cluster we have a different dominant religion (it’s a good sign!). It’s logical to have two times, two religions, because we have nine clusters and eight religions. After, we decide to investigate the percentage of the continents for each cluster (see in Figure 17 – Continents % per Cluster). I would like to mention that we didn’t take into consideration the
continents attributes in the clusters creation, because we base on flag characteristics.

![alt text](https://github.com/ggeop/Flag-Study/blob/master/figures/figure_17.PNG)

If we summarize the above pies we have the following table (see in Table 5 -Clusters Summary Data)

![alt text](https://github.com/ggeop/Flag-Study/blob/master/tables/table_5.PNG)

In the next table (see in Table 6 - Flags Sample for each Cluster) we have randomly sample 4 flags for each cluster in order to have a visual overview of the clusters (see in Appendix I, for an analytical overview of each cluster, Table 10 – Table 18). For example in the second cluster we have flags with 3 dominant colours which have the green as the most dominant. 

![alt text](https://github.com/ggeop/Flag-Study/blob/master/tables/table_6.PNG)

Finally, we have createD tables with all the flags of each cluster (see in Appendix I, the Table 7 - Countries of Clusters 1,2, 3 Table 8 - Countries of Clusters 4,5,6 Table 9 - Countries of Clusters 7,8,9)

## Conclusions
In the first part of the project we had to find a method to classify the nation’s flags in that way that we can identify the religion of the country based on their characteristics. We found that the best method was Classification Tree with Rpart library. Rpart package is a CART partitioning tree with different implementation. The Rpart programs build classification or regression models of a very general structure using a two stage procedure; the resulting models can be represented as binary trees. An example is some preliminary data gathered at Stanford on revival of cardiac arrest patients by paramedics.

In the second part of the project we had to create clusters of flags with common characteristics. At the beginning, in this method, we had to find the optimum number of clusters and then to assign each flag to one of them. We tried two methods (Hierarchical Clustering and PAM Method) and we evaluated the clusters by using Silhouette plots. Finally, we chose the PAM method as better for our occasion.

There are several distinctive characteristics by which we could classify better the flags. For example:

o	Color separation types (e.g. two colours, tree colours)
o	Flag shapes
o	Predominant colour of the flag (Mainhue colour)
o	Color symbolism (the same colour in different countries has different meaning)
o	Symbols of the flags (for example, a cross, a star, a crescent)

So, we could have the following more clear clusters:

o	Cluster 1 - One colour flags with a symbol in the center
o	Cluster 2 - Two colour flags which divided into horizontal and vertical
o	Cluster 3 - Three colour vertical stripes
o	Cluster 4 - Three colour horizontal stripes
o	Cluster 5 - Flags with diagonial division
o	Cluster 6 - Flags which are relatively symmetrical
o	Cluster 7 - Canton flags, these flags have one or two colours except the canton, e.g Greek flag
o	Cluster 8 - Flags with vertical stripes at the flagpole
o	Cluster 9 - Flags with the Scandinavian cross
o	Cluster 10 - Flags with a separating triangle
o	Cluster 11 - Multicolor flags
o	Cluster 12 - Other flags, that are not in the above clusters
