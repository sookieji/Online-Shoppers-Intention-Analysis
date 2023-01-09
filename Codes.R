#Group Project

shoppers.df <- read.csv('online_shoppers_intention.csv')
shoppers.df <- shoppers.df[ , -c(12:15)]  
#we don't know what the different categories in these four variables stand for(OperatingSystem, Browser, Region, TrafficType)

############################ Data Preprocessing ################################
# check if there is missing value
sapply(shoppers.df, function(x) sum(is.na(x))) 

# turn logic variables into binary
str(shoppers.df)
shoppers.df$Revenue <- ifelse(shoppers.df$Revenue==TRUE, 1, 0)
shoppers.df$Weekend <- ifelse(shoppers.df$Weekend==TRUE, 1, 0)
str(shoppers.df)


######################### Cluster Analysis kmeans ##############################
# turning character variables into numerical variables to calculate distances
shoppers_cluster.df <- shoppers.df
sapply(shoppers_cluster.df, typeof)
unique(shoppers_cluster.df$Month)
unique(shoppers_cluster.df$VisitorType)
shoppers_cluster.df$Month <- as.numeric(factor(shoppers_cluster.df$Month, order = TRUE, levels =c('Feb', 'Mar', 'May', 'June','Jul', 'Aug', 'Sep','Oct', 'Nov','Dec')))
shoppers_cluster.df$VisitorType <- as.numeric(factor(shoppers_cluster.df$VisitorType, order = TRUE, levels =c('Returning_Visitor', 'Other', 'New_Visitor')))
str(shoppers_cluster.df)

# normalize input variables
shoppers_cluster.df.norm <- sapply(shoppers_cluster.df, scale)   
set.seed(2) 
km <- kmeans(shoppers_cluster.df.norm, 6)
km$centers
km$size

# plot clusters
plot(c(0), xaxt = 'n', ylab = "", type = "l", 
     ylim = c(min(km$centers), max(km$centers)), xlim = c(0, 14))
axis(1, at = c(1:14), labels = names(shoppers_cluster.df))
for (i in c(1:6))
  lines(km$centers[i,], lty = i, lwd = 3, col = ifelse(i %in% c(1, 3, 5),
                                                       "black", "dark grey"))
text(x =0.5, y = km$centers[, 1], labels = paste("Cluster", c(1:6)))



############################## Supervised learning #############################

# Sampling train dataset & validation dataset
library(dplyr)
set.seed(2)
#take a sample of validation dataset before drawing the undersampling
valid.index <- sample(c(1:dim(shoppers.df)[1]), dim(shoppers.df)[1]*0.2)
valid.df <- shoppers.df[valid.index, ]
table(valid.df$Revenue) 

#undersampling for training dataset
train_original.df <- shoppers.df[-valid.index, ]
train.true.revenue.df <- train_original.df[train_original.df$Revenue==1, ]                        #all revenue True records are picked
false.revenue.df <- train_original.df[train_original.df$Revenue==0, ]
train.false.revenue.index <- sample(c(1:dim(false.revenue.df)[1]), dim(train.true.revenue.df)[1]) #randomly pick revenue False records as many as True
train.false.revenue.df <- false.revenue.df[train.false.revenue.index, ]
train.df <- rbind(train.true.revenue.df, train.false.revenue.df)                                  #merge two subsets to form the balanced train.df
table(train.df$Revenue)



############################### Decision tree ##################################
library(rpart)
library(rpart.plot)
library(caret)

#default tree
default.ct <- rpart(Revenue ~ ., data = train.df, method = "class")
prp(default.ct, type = 1, extra = 2, under = TRUE, split.font = 1, varlen = -10)
length(default.ct$frame$var[default.ct$frame$var == "<leaf>"])

#pruned tree, to get the number of nodes with best perfomance
set.seed(1)
cv.ct <- rpart(Revenue ~ ., data = train.df, method = "class", cp = 0.00001, minsplit = 1, xval = 5)  
printcp(cv.ct) 
pruned.ct <- prune(cv.ct, cp = 0.00226684) #choose the CP value with the lowest misclassification rate
printcp(pruned.ct)
prp(pruned.ct, type = 1, extra = 1, under = TRUE, split.font = 1, varlen = -10, 
    box.col=ifelse(pruned.ct$frame$var == "<leaf>", 'gray', 'white')) 

#default model accuracy rate: 0.8406
default.ct.pred.valid <- predict(default.ct, valid.df, type = "class")
confusionMatrix(default.ct.pred.valid, as.factor(valid.df$Revenue))
#pruned tree accuracy rate: 0.837
pruned.ct.pred.valid <- predict(pruned.ct, valid.df, type = "class")
confusionMatrix(pruned.ct.pred.valid, as.factor(valid.df$Revenue))


#randomForest method ends up with optimal accuracy rate within decision tree
library(randomForest)
rf <- randomForest(as.factor(Revenue) ~ ., data = train.df, ntree = 500, 
                   mtry = 4, nodesize = 5, importance = TRUE)  
summary(rf)
#randomForest accuracy rate: 0.852
rf.pred <- predict(rf, valid.df)
confusionMatrix(rf.pred, as.factor(valid.df$Revenue))
# variable importance plot
varImpPlot(rf, type = 1, main = 'Variable Importance Plot')



############################## Neural Networks #################################
#Notes: all variables need to be numerical when running NN (no need for decision tree & regression)
library(neuralnet)
library(plyr)
set.seed(1)  

#turning character variables into numerical variables, same codes as that used in the clustering
train_nn.df <- train.df
valid_nn.df <- valid.df
train_nn.df$Month <- as.numeric(factor(train_nn.df$Month, order = TRUE, levels =c('Feb', 'Mar', 'May', 'June','Jul', 'Aug', 'Sep','Oct', 'Nov','Dec')))
valid_nn.df$Month <- as.numeric(factor(valid_nn.df$Month, order = TRUE, levels =c('Feb', 'Mar', 'May', 'June','Jul', 'Aug', 'Sep','Oct', 'Nov','Dec')))


#run NN with 5 most important variables
nn <- neuralnet(Revenue ~ ProductRelated + ProductRelated_Duration + ExitRates + PageValues + Month, 
                data = train_nn.df, linear.output = F, hidden = 3, learningrate = 0.01, stepmax =1e9)

plot(nn, rep="best") 
nn$weights

#NN accuracy rate: 0.8698
prediction(nn)
nn.pred <- predict(nn, valid_nn.df, type = "response")
nn.pred.classes <- ifelse(nn.pred > 0.5, 1, 0)
confusionMatrix(as.factor(nn.pred.classes), as.factor(valid_nn.df$Revenue))



############################# Logistic Regression ##############################
logit.reg <- glm(Revenue ~ ., data = train.df, family = "binomial") 
options(scipen=999)
summary(logit.reg)

# logistic regression accuracy rate: 0.9019
logit.reg.pred <- predict(logit.reg, valid.df, type = "response")
logit.reg.pred.classes <- ifelse(logit.reg.pred > 0.8, 1, 0) #I tried manually the threshold of 0.8 comes with best accuracy rate
confusionMatrix(as.factor(logit.reg.pred.classes), as.factor(valid.df$Revenue)) 

# model selection
full.logit.reg <- glm(Revenue ~ ., data = train.df, family = "binomial") 
backwards = step(full.logit.reg)
summary(backwards)
formula(backwards)
# after model selection accuracy rate: 0.9011
backwards.reg.pred <- predict(backwards, valid.df, type = "response")
backwards.reg.pred.classes <- ifelse(backwards.reg.pred > 0.8, 1, 0)
confusionMatrix(as.factor(backwards.reg.pred.classes), as.factor(valid.df$Revenue))


########################## Classifier Evaluation ###############################
# ROC curve to compares 3 different classification models using validation data.
# random forest #Area under the curve: 0.8462
library(pROC)
forest.df <- data.frame(actual = valid.df$Revenue, prob = rf.pred)
forest.r <- roc(forest.df$actual, predictor = factor(forest.df$prob, ordered = TRUE))
plot.roc(forest.r)
auc(forest.r)


# NN Area under the curve: 0.877
nn.df <- data.frame(actual = valid_nn.df$Revenue, prob = nn.pred)
nn.r <- roc(nn.df$actual, nn.df$prob)
plot.roc(nn.r)
auc(nn.r)


#logit model #Area under the curve: 0.9099
logit.df <- data.frame(actual = valid.df$Revenue, prob = logit.reg.pred)
logit.df <- logit.df[order(logit.df$prob, decreasing = TRUE), ]
logit.r <- roc(logit.df$actual, logit.df$prob)
plot.roc(logit.r)
auc(logit.r)


##############OPTIONAL
# without undersampling, using original train dataset
# to check if undersampling improve model accuracy
original.lr <- glm(Revenue ~ ., data = train_original.df, family = "binomial") 
summary(original.lr)
original.lr.pred <- predict(original.lr, valid.df, type = "response")
original.lr.pred.classes <- ifelse(original.lr.pred > 0.8, 1, 0)
confusionMatrix(as.factor(original.lr.pred.classes), as.factor(valid.df$Revenue))
