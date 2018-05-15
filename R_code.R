###########################
#Libraries
###########################
library('mclust')
library('cluster')
library('factoextra')
library('plotly')
library('plyr')
library('tree')
library('caret')
library('randomForest')
library('rpart')
library('e1071')
library("Rtsne")

###########################
#Import Dataset
###########################
flag <- read.csv("C:/Users/georg/Dropbox/Business Analytics/Winter-Quarter/Statistics_2/Assignment_2/flag.data", header=FALSE)

####################################################################################
################################ 1. Classiffication  ###############################
####################################################################################

#Religion
flag$V7<- as.factor(flag$V7)
flag$V7<-revalue(flag$V7, c("0"="Catholic", "1"="Other Christian", "2"="Muslim", "3"="Buddhist", "4"="Hindu", "5"="Ethnic","6"="Marxist","7"="Others"))

#We use only the characteristics of the flags
flag_class<-flag[,c(7:30)]

#Rename attributes
names(flag_class) <- c("religion","bars","stripes","colours","red","green","blue","gold","white","black","orange","mainhue","circles","crosses","saltires","quarters","sunstars","crescent","triangle","icon","animate","text","topleft","botright")

flag_num<-flag_class

flag_class$religion<-as.factor(flag_class$religion)
flag_class$religion<-revalue(flag_class$religion, c("0"="Catholic", "1"="Other Christian", "2"="Muslim", "3"="Buddhist", "4"="Hindu", "5"="Ethnic","6"="Marxist","7"="Others"))

#Change the types of the characteristics

##Factors
flag_class$religion<-as.factor(flag_class$religion)
flag_class$red<-as.factor(flag_class$red)
flag_class$green<-as.factor(flag_class$green)
flag_class$blue<-as.factor(flag_class$blue)
flag_class$gold<-as.factor(flag_class$gold)
flag_class$white<-as.factor(flag_class$white)
flag_class$black<-as.factor(flag_class$black)
flag_class$orange<-as.factor(flag_class$orange)
flag_class$crescent<-as.factor(flag_class$crescent)
flag_class$triangle<-as.factor(flag_class$triangle)
flag_class$animate<-as.factor(flag_class$animate)
flag_class$text<-as.factor(flag_class$text)
flag_class$topleft<-as.factor(flag_class$topleft)
flag_class$botright<-as.factor(flag_class$botright)

##Integers
flag_class$bars<-as.integer(flag_class$bars)
flag_class$stripes<-as.integer(flag_class$stripes)
flag_class$colours<-as.integer(flag_class$colours)
flag_class$circles<-as.integer(flag_class$circles)
flag_class$crosses<-as.integer(flag_class$crosses)
flag_class$saltires<-as.integer(flag_class$saltires)
flag_class$quarters<-as.integer(flag_class$quarters)
flag_class$sunstars<-as.integer(flag_class$sunstars)

str(flag_class)


###########################
#Descriptive Analysis-Plots
###########################

####Religions
#Sort the religions (for better visualization)
V7<-as.data.frame(table(flag$V7))
V7<-V7[order(V7$Freq, decreasing = TRUE),]
V7$Var1 <- factor(V7$Var1, levels = unique(V7$Var1)[order(V7$Freq, decreasing = TRUE)])

p7<-plot_ly(V7,x=~Var1,
        y=~Freq, type = 'bar', 
        marker = list(color ='rgb(158,202,225)',
                      line = list(color = 'rgb(255,255,255)', 
                                  width = 6.0)
        )) %>% layout(title = "Religions and Flags",
                      xaxis = list(title = "Religions"),
                      yaxis = list(title = "Number of Flags"))

####Bars
V8<-as.data.frame(table(flag$V8))
V8<-V8[order(V8$Freq, decreasing = TRUE),]
V8$Var1 <- factor(V8$Var1, levels = unique(V8$Var1)[order(V8$Freq, decreasing = TRUE)])

p8<-plot_ly(V8,x=~Var1,
        y=~Freq, type = 'bar', 
        marker = list(color ='rgb(158,202,225)',
                      line = list(color = 'rgb(255,255,255)', 
                                  width = 6.0)
        )) %>% layout(title = "Vertical Number of bars",
                      xaxis = list(title = "Bars"),
                      yaxis = list(title = "Number of flags"))

####Stripes
V9<-as.data.frame(table(flag$V9))
V9<-V9[order(V9$Freq, decreasing = TRUE),]
V9$Var1 <- factor(V9$Var1, levels = unique(V9$Var1)[order(V9$Freq, decreasing = TRUE)])

p9<-plot_ly(V9,x=~Var1,
        y=~Freq, type = 'bar', 
        marker = list(color ='rgb(158,202,225)',
                      line = list(color = 'rgb(255,255,255)', 
                                  width = 6.0)
        )) %>% layout(title = "Number of Stripes",
                      xaxis = list(title = "Stripes"),
                      yaxis = list(title = "Number of flags"))

####Colours
V10<-as.data.frame(table(flag$V10))
V10<-V10[order(V10$Freq, decreasing = TRUE),]
V10$Var1 <- factor(V10$Var1, levels = unique(V10$Var1)[order(V10$Freq, decreasing = TRUE)])

p10<-plot_ly(V10,x=~Var1,
        y=~Freq, type = 'bar', 
        marker = list(color ='rgb(158,202,225)',
                      line = list(color = 'rgb(255,255,255)', 
                                  width = 6.0)
        )) %>% layout(title = "Number of Different Colours",
                      xaxis = list(title = "Number of different colours"),
                      yaxis = list(title = "Number of flags"))



####Main Hue
V18<-as.data.frame(table(flag$V18))
V18<-V18[order(V18Freq, decreasing = TRUE),]
V18$Var1 <- factor(V18$Var1, levels = unique(V18$Var1)[order(V18$Freq, decreasing = TRUE)])

p18<-plot_ly(V18,x=~Var1,
             y=~Freq, type = 'bar', 
             marker = list(color ='rgb(158,202,225)',
                           line = list(color = 'rgb(255,255,255)', 
                                       width = 6.0)
             )) %>% layout(
                           title = "Main Hue Color",
                           xaxis = list(title = "Colours"),
                           yaxis = list(title = "Number of flags"))



####Circles
V19<-as.data.frame(table(flag$V19))
V19<-V19[order(V19Freq, decreasing = TRUE),]
V19$Var1 <- factor(V19$Var1, levels = unique(V19$Var1)[order(V19$Freq, decreasing = TRUE)])

p19<-plot_ly(V19,x=~Var1,
        y=~Freq, type = 'bar', 
        marker = list(color ='rgb(158,202,225)',
                      line = list(color = 'rgb(255,255,255)', 
                                  width = 6.0)
        )) %>% layout(title = "Number of Cyrcles",
                      xaxis = list(title = "Number of Circles"),
                      yaxis = list(title = "Number of flags"))


####Number of Upright Crosses
V20<-as.data.frame(table(flag$V20))
V20<-V20[order(V20Freq, decreasing = TRUE),]
V20$Var1 <- factor(V20$Var1, levels = unique(V20$Var1)[order(V20$Freq, decreasing = TRUE)])

p20<-plot_ly(V20,x=~Var1,
        y=~Freq, type = 'bar', 
        marker = list(color ='rgb(158,202,225)',
                      line = list(color = 'rgb(255,255,255)', 
                                  width = 6.0)
        )) %>% layout(title = "Number of Upright Crosses",
                      xaxis = list(title = "Number of Crosses"),
                      yaxis = list(title = "Number of flags"))

####Number of saltires
V21<-as.data.frame(table(flag$V21))
V21<-V21[order(V21Freq, decreasing = TRUE),]
V21$Var1 <- factor(V21$Var1, levels = unique(V21$Var1)[order(V21$Freq, decreasing = TRUE)])

p21<-plot_ly(V21,x=~Var1,
        y=~Freq, type = 'bar', 
        marker = list(color ='rgb(158,202,225)',
                      line = list(color = 'rgb(255,255,255)', 
                                  width = 6.0)
        )) %>% layout(title = "Number of Saltires",
                      xaxis = list(title = "Number of Diagonial Crosses"),
                      yaxis = list(title = "Number of flags"))


####Number of Quarters
V22<-as.data.frame(table(flag$V22))
V22<-V22[order(V22Freq, decreasing = TRUE),]
V22$Var1 <- factor(V22$Var1, levels = unique(V22$Var1)[order(V22$Freq, decreasing = TRUE)])

p22<-plot_ly(V22,x=~Var1,
             y=~Freq, type = 'bar', 
             marker = list(color ='rgb(158,202,225)',
                           line = list(color = 'rgb(255,255,255)', 
                                       width = 6.0)
             )) %>% layout(title = "Number of Quarters",
                           xaxis = list(title = "Quarters"),
                           yaxis = list(title = "Number of flags"))

###########################
#Tree Model (all variables)
###########################
#Create the model
fit1<-tree(flag_class$religion~.,data=flag_class,split="gini")

#Visualize the model
plot(fit1, main="Initial Classification Tree"); text(fit1, cex=0.8, col="dodgerblue3")

#k-Fold Validation
re<-NULL
deiktes<-sample(1:nrow(flag_class))

for (k in c(1,2,3,4,6,8,10)) {
  print(k)
  
  omades<- as.integer(nrow(flag_class)/k)
  
  t<-NULL
  
  for (i in 1:omades) {
    te<- deiktes[ ((i-1)*k+1):(i*k)]
    train <- flag_class[-te,]
    test <-   flag_class[te,]
    cl <- factor(flag_class$religion[-te])
    z <- tree(cl~.,data=train,split="gini")
    pr<-  predict(z, test,type='class')
    t<- c(t,   sum(flag_class$religion[te] == pr)    /dim(test)[1])
  }
  
  
  re<-c(re,mean(t))
}

#Avg final score
mean(re) #0.42

########################################################
#Tree Model (with selected variables from random forest)
########################################################

#######################################
#Variables Selection with Random Forest
#######################################

#Random Forest
rf <- randomForest(religion~.,data=flag_class, trials=100, method='class')

#The general idea is to permute the values of each feature
#and measure how much the permutation decreases the 
#accuracy of the model

#Initialize for the search

importance<-rf$importance
importance<-importance[order(importance),]

not.them<-NULL
accur_score<-0
re<-0
p<-1

while (accur_score<=mean(re)){

  #Create the list with the variables that we minimize the accuracy of the model
  not.them<-c(not.them, importance[p])
  accur_score<-mean(re)
  re<-NULL
  deiktes<-sample(1:nrow(flag_class))
  
  #Cross validation
  for (k in c(1,2,3,4,6,8,10,15,20)) {
    
    omades<- as.integer(nrow(flag_class)/k)
    
    t<-NULL
    
    for (i in 1:omades) {
      te<- deiktes[ ((i-1)*k+1):(i*k)]
      train <- flag_class[-te,]
      test <-   flag_class[te,]
      cl <- factor(flag_class$religion[-te])
      z <- tree(cl~.,data=train[,colnames(train)!=not.them],split="gini")
      pr<-  predict(z, test,type='class')
      t<- c(t,   sum(flag_class$religion[te] == pr)    /dim(test)[1])
    }
    
    
    re<-c(re,mean(t))
  }
  
print(p)
p<-p+1

}

print(not.them)
print(accur_score)

#Create the model
fit3<-tree(flag_class$religion~.,data=flag_class[,colnames(train)!=not.them],split="deviance")

#Remove the less important variables
varImpPlot(rf,  sort = T,n.var=ncol(flag_class)-length(not.them)-1, main=" Selected Variables Importance")


#Visualize the model
plot(fit3); text(fit3, cex=0.8, col="dodgerblue3")


##########################################
#Tree model (with rpart Library + pruning)
##########################################

fit2<-rpart(flag_class$religion~.,data=flag_class)
pfit<- prune(fit2, cp=fit2$cptable[which.min(fit2$cptable[,"xerror"]),"CP"])


#Visualize prune tree
plot(fit2); text(fit2, cex=0.8, col="dodgerblue3") #Before pruning
plot(pfit); text(pfit, cex=0.8, col="dodgerblue3") #After pruning

#k-Fold Validation
re<-NULL
deiktes<-sample(1:nrow(flag_class))

for (k in c(1,2,3,4,6,8,10,15,20)) {
  print(k)
  
  omades<- as.integer(nrow(flag_class)/k)
  
  t<-NULL
  
  for (i in 1:omades) {
    te<- deiktes[ ((i-1)*k+1):(i*k)]
    train <- flag_class[-te,]
    test <-   flag_class[te,]
    cl <- factor(flag_class$religion[-te])
    fit2<-rpart(cl~.,data=train)
    z<- prune(fit2, cp=fit2$cptable[which.min(fit2$cptable[,"xerror"]),"CP"])
    pr<-  predict(z, test,type='class')
    t<- c(t,   sum(flag_class$religion[te] == pr)    /dim(test)[1])
  }
  
  
  re<-c(re,mean(t))
}

#Avg final score
mean(re)


###############################
#SVM
###############################

####################
#### Attention! ####
####################
#The tune method takes more than 20 minutes to finish!
#Tune the parameters
obj = tune.svm(religion~.,data=flag_class,cost=10:100,gamma=seq(0,3,0.1)) 

#Train the model
svm_fit<-svm(religion ~ ., data = flag_class, cost = 19,cross=10, gamma = 0.2)


#k-Fold Validation
re<-NULL
accur<-NULL

deiktes<-sample(1:nrow(flag_class))

for (k in c(1,2,3,4,6,8,10,15,20)) {
  print(k)
  
  omades<- as.integer(nrow(flag_class)/k)
  
  t<-NULL
  
  for (i in 1:omades) {
    te<- deiktes[ ((i-1)*k+1):(i*k)]
    train <- flag_class[-te,]
    test <-   flag_class[te,]
    cl <- factor(flag_class$religion[-te])
    svm_fit <- svm(cl ~ ., data = train, cost = 19,cross=10, gamma = 0.2)
    pr<-  predict(svm_fit, test,type='class')
    t<- c(t,   sum(flag_class$religion[te] == pr)    /dim(test)[1])
    
    
    ## compute svm confusion matrix & Accuracy
 
    matrix<-table(pred=pr, true=flag_class$religion[te])
    
    accuracy<-sum(diag(matrix))/sum(matrix)
  }
  
  
  re<-c(re,mean(t))
  accur<-c(accur,accuracy)
}

#Avg final score
mean(re) #0.76
mean(accur) #0.67

####################################################################################
################################## 2. Clustering ###################################
####################################################################################

#We use only the characteristics of the flags
flag_clust<-flag[,c(7:30)]

#Rename the Features
names(flag_clust) <- c("religion","bars","stripes","colours","red","green","blue","gold","white","black","orange","mainhue","circles","crosses","saltires","quarters","sunstars","crescent","triangle","icon","animate","text","topleft","botright")



#calculate the distance
gower_dist <- daisy(flag_clust[,-1],
                    metric = "gower"
                    )

summary(gower_dist)

#  Output most similar pair
gower_mat <- as.matrix(gower_dist)

flag_clust[
  which(gower_mat == min(gower_mat[gower_mat != min(gower_mat)]),
        arr.ind = TRUE)[1, ],]
#The results are the flags 85, 168 which are the flags of Syria and Iraq

#  Output most dissimilar pair
gower_mat <- as.matrix(gower_dist)

flag_clust[
  which(gower_mat == max(gower_mat[gower_mat != max(gower_mat)]),
        arr.ind = TRUE)[1, ], ]

#The results are the flags 77, 79 which are the flags of Haiti and Hong-Kong

################################################
#Calculate silhouette width for many k using PAM
################################################

sil_width <- c(NA)

for(i in 2:10){
  
  pam_fit <- pam(gower_dist,
                 diss = TRUE,
                 k = i)
  
  sil_width[i] <- pam_fit$silinfo$avg.width
  
}

# Plot sihouette width (higher is better)

plot(1:10, sil_width,
     xlab = "Number of clusters",
     ylab = "Silhouette Width", col="blue")
lines(1:10, sil_width)

########################
#Hierarchical Clustering
########################

# Euclidean Distance & Ward Linkage
gower_ward <- hclust(gower_dist,method='ward.D')

#Visualization
plot(gower_ward)
rect.hclust(gower_ward, k = 9, border = "red")
summary(gower_ward)

clusterCut <- cutree(gower_ward, 9)

table(actual = flag_clust$V7, predicted = clusterCut)

fviz_silhouette(silhouette(clusterCut,gower_dist ))  

########################
#PAM Method
########################

pam_fit <- pam(gower_dist, diss = TRUE, k = 9)

#Summarys statistics for each cluster
pam_results <- flag_clust %>%
  dplyr::select(- religion) %>%
  mutate(cluster = pam_fit$clustering) %>%
  group_by(cluster) %>%
  do(the_summary = summary(.))

pam_results$the_summary

###########################
#Clustering visualization
###########################
tsne_obj <- Rtsne(gower_dist, is_distance = TRUE)

tsne_data <- tsne_obj$Y %>%
  data.frame() %>%
  setNames(c("X", "Y")) %>%
  mutate(cluster = factor(pam_fit$clustering),
         name = flag_clust$V7)

ggplot(aes(x = X, y = Y), data = tsne_data) +
  geom_point(aes(color = cluster))
    

plot(pam_fit, main="Silhouette Plot")
clusplot(pam_fit)

#Count the number of observation for each cluster
table(pam_fit$clustering)

fviz_silhouette(pam_fit)

pam_fit$clustering

plot_ly( x = ~flag$V19, y = ~flag_class$colours, color = ~as.factor(pam_fit$clustering),type = 'bar')

plot_ly( x = ~flag$V19, y = ~flag_class$colours, color = ~as.factor(pam_fit$clustering),type = 'bar')

#Add a new column with cluster ID
flag_clust$clusters<-as.factor(pam_fit$clustering)

names(flag_clust) <- c("religion","bars","stripes","colours","red","green","blue","gold","white","black","orange","mainhue","circles","crosses","saltires","quarters","sunstars","crescent","triangle","icon","animate","text","topleft","botright", "clusters")

###########################
#Religion Pie
###########################

#Create a prop table for all the clusters
religion_table<-prop.table(table(flag_clust$clusters, flag_clust$religion),2)
plot(religion_table)

#Cluster 1
plot_ly( labels = ~names(religion_table[1,]),
         values = ~religion_table[1,],
         textinfo = 'label+percent',
         insidetextfont = list(color = '#FFFFFF'),
         marker = list(line = list(color = '#FFFFFF', width = 1)),
         type = 'pie')%>%
  layout(title = 'Religions % in Cluster 1')

#Cluster 2
plot_ly( labels = ~names(religion_table[2,]),
         values = ~religion_table[2,],
         textinfo = 'label+percent',
         insidetextfont = list(color = '#FFFFFF'),
         marker = list(line = list(color = '#FFFFFF', width = 1)),
         type = 'pie')%>%
  layout(title = 'Religions % in Cluster 2')

#Cluster 3
plot_ly( labels = ~names(religion_table[3,]),
         values = ~religion_table[3,],
         textinfo = 'label+percent',
         insidetextfont = list(color = '#FFFFFF'),
         marker = list(line = list(color = '#FFFFFF', width = 1)),
         type = 'pie')%>%
  layout(title = 'Religions % in Cluster 3')

#Cluster 4
plot_ly( labels = ~names(religion_table[4,]),
         values = ~religion_table[4,],
         textinfo = 'label+percent',
         insidetextfont = list(color = '#FFFFFF'),
         marker = list(line = list(color = '#FFFFFF', width = 1)),
         type = 'pie')%>%
  layout(title = 'Religions % in Cluster 4')

#Cluster 5
plot_ly( labels = ~names(religion_table[5,]),
         values = ~religion_table[5,],
         textinfo = 'label+percent',
         insidetextfont = list(color = '#FFFFFF'),
         marker = list(line = list(color = '#FFFFFF', width = 1)),
         type = 'pie')%>%
  layout(title = 'Religions % in Cluster 5')


#Cluster 6
plot_ly( labels = ~names(religion_table[6,]),
         values = ~religion_table[6,],
         textinfo = 'label+percent',
         insidetextfont = list(color = '#FFFFFF'),
         marker = list(line = list(color = '#FFFFFF', width = 1)),
         type = 'pie')%>%
  layout(title = 'Religions % in Cluster 6')


#Cluster 7
plot_ly( labels = ~names(religion_table[7,]),
         values = ~religion_table[7,],
         textinfo = 'label+percent',
         insidetextfont = list(color = '#FFFFFF'),
         marker = list(line = list(color = '#FFFFFF', width = 1)),
         type = 'pie')%>%
  layout(title = 'Religions % in Cluster 7')


#Cluster 8
plot_ly( labels = ~names(religion_table[8,]),
         values = ~religion_table[8,],
         textinfo = 'label+percent',
         insidetextfont = list(color = '#FFFFFF'),
         marker = list(line = list(color = '#FFFFFF', width = 1)),
         type = 'pie')%>%
  layout(title = 'Religions % in Cluster 8')

#Cluster 9
plot_ly( labels = ~names(religion_table[9,]),
         values = ~religion_table[9,],
         textinfo = 'label+percent',
         insidetextfont = list(color = '#FFFFFF'),
         marker = list(line = list(color = '#FFFFFF', width = 1)),
         type = 'pie')%>%
  layout(title = 'Religions % in Cluster 9')



###########################
# Continent Pies (landmass)
###########################

#Create a prop table for all the clusters
flag_clust$landmass<-as.factor(flag$V2)
flag_clust$landmass<- revalue(flag_clust$landmass, c( '1'='N.America', '2'='S.America', '3'='Europe', '4'='Africa', '5'='Asia', '6'='Oceania'))
religion_table<-prop.table(table(flag_clust$clusters, flag_clust$landmass),2)
plot(religion_table)

#Cluster 1
plot_ly( labels = ~names(religion_table[1,]),
         values = ~religion_table[1,],
         textinfo = 'label+percent',
         insidetextfont = list(color = '#FFFFFF'),
         marker = list(line = list(color = '#FFFFFF', width = 1)),
         type = 'pie')%>%
  layout(title = 'Religions % in Cluster 1')

#Cluster 2
plot_ly( labels = ~names(religion_table[2,]),
         values = ~religion_table[2,],
         textinfo = 'label+percent',
         insidetextfont = list(color = '#FFFFFF'),
         marker = list(line = list(color = '#FFFFFF', width = 1)),
         type = 'pie')%>%
  layout(title = 'Religions % in Cluster 2')

#Cluster 3
plot_ly( labels = ~names(religion_table[3,]),
         values = ~religion_table[3,],
         textinfo = 'label+percent',
         insidetextfont = list(color = '#FFFFFF'),
         marker = list(line = list(color = '#FFFFFF', width = 1)),
         type = 'pie')%>%
  layout(title = 'Religions % in Cluster 3')


#Cluster 4
plot_ly( labels = ~names(religion_table[4,]),
         values = ~religion_table[4,],
         textinfo = 'label+percent',
         insidetextfont = list(color = '#FFFFFF'),
         marker = list(line = list(color = '#FFFFFF', width = 1)),
         type = 'pie')%>%
  layout(title = 'Religions % in Cluster 4')



#Cluster 5
plot_ly( labels = ~names(religion_table[5,]),
         values = ~religion_table[5,],
         textinfo = 'label+percent',
         insidetextfont = list(color = '#FFFFFF'),
         marker = list(line = list(color = '#FFFFFF', width = 1)),
         type = 'pie')%>%
  layout(title = 'Religions % in Cluster 5')


#Cluster 6
plot_ly( labels = ~names(religion_table[6,]),
         values = ~religion_table[6,],
         textinfo = 'label+percent',
         insidetextfont = list(color = '#FFFFFF'),
         marker = list(line = list(color = '#FFFFFF', width = 1)),
         type = 'pie')%>%
  layout(title = 'Religions % in Cluster 6')


#Cluster 7
plot_ly( labels = ~names(religion_table[7,]),
         values = ~religion_table[7,],
         textinfo = 'label+percent',
         insidetextfont = list(color = '#FFFFFF'),
         marker = list(line = list(color = '#FFFFFF', width = 1)),
         type = 'pie')%>%
  layout(title = 'Religions % in Cluster 7')


#Cluster 8
plot_ly( labels = ~names(religion_table[8,]),
         values = ~religion_table[8,],
         textinfo = 'label+percent',
         insidetextfont = list(color = '#FFFFFF'),
         marker = list(line = list(color = '#FFFFFF', width = 1)),
         type = 'pie')%>%
  layout(title = 'Religions % in Cluster 8')


#Cluster 9
plot_ly( labels = ~names(religion_table[9,]),
         values = ~religion_table[9,],
         textinfo = 'label+percent',
         insidetextfont = list(color = '#FFFFFF'),
         marker = list(line = list(color = '#FFFFFF', width = 1)),
         type = 'pie')%>%
  layout(title = 'Religions % in Cluster 9')

################################
#Countriers of each cluster
################################

#Create a prop table for all the clusters
flag_clust$name<-as.factor(flag$V1)
countries_table<-table(flag_clust$name, flag_clust$clusters)

#Cluster 1 countries
View(countries_table[countries_table[,1]==1,1])

#Cluster 2 countries
View(countries_table[countries_table[,2]==1,2])

#Cluster 3 countries
View(countries_table[countries_table[,3]==1,3])

#Cluster 4 countries
View(countries_table[countries_table[,4]==1,4])

#Cluster 5 countries
View(countries_table[countries_table[,5]==1,5])

#Cluster 6 countries
View(countries_table[countries_table[,6]==1,6])

#Cluster 7 countries
View(countries_table[countries_table[,7]==1,7])

#Cluster 8 countries
View(countries_table[countries_table[,8]==1,8])

#Cluster 9 countries
View(countries_table[countries_table[,9]==1,9])
