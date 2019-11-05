#Chapter 8, Group 4, Marco Duran Perez, Hilary Bali, George Garcia

# Import packages and data

library(mlbench)
library(caret)
library(caTools)
library(AppliedPredictiveModeling)
library(rpart)
library(partykit)
library(RWeka)
library(randomForest)
library(doMC)
library(party)
library(gbm)
library(cubist)
data(BostonHousing)

# Preprocess the predictors using Yeo Johnson and Spatial Sign
BostonHousingYJ <- preProcess(BostonHousing[, -4], method = "YeoJohnson")
yjData <- predict(BostonHousingYJ, newdata = BostonHousing[, -4])
BHtrans <- as.data.frame(spatialSign(yjData))


# Create training/testing data
set.seed(111)
sample <- sample.split(BHtrans,SplitRatio = 0.70)
train1 <- subset(BHtrans, sample == TRUE)
test1 <- subset(BHtrans, sample == FALSE)

# Create a control function
set.seed(111)
indx <- createFolds(train1, returnTrain = TRUE)
ctrl <- trainControl(method = "cv", index = indx)

# Conditional Inference Tree
trainData <- train1[, -13]
trainData$y <- train1[, 13]

rpStump <- rpart(y ~ ., data = trainData, 
                 control = rpart.control(maxdepth = 1))
rpSmall <- rpart(y ~ ., data = trainData, 
                 control = rpart.control(maxdepth = 2))

set.seed(111)
cartTune <- train(x = train1[, -13], y = train1[, 13],
                  method = "rpart",
                  tuneLength = 25,
                  trControl = ctrl)
cartTune #RMSE = .5415097
plot(cartTune, scales = list(x = list(log = 10)))
cartImp <- varImp(cartTune, scale = FALSE, competes = FALSE)
cartImp

### Save the test set results in a data frame                 
testResults <- data.frame(obs = train1[, 13],
                          CART = predict(cartTune, train1[, -13]))

### Tune the conditional inference tree

cGrid <- data.frame(mincriterion = sort(c(.95, seq(.75, .99, length = 2))))

set.seed(111)
ctreeTune <- train(x = train1[, -13], y = train1[, 13],
                   method = "ctree",
                   tuneGrid = cGrid,
                   trControl = ctrl)
ctreeTune
plot(ctreeTune)

##ctreeTune$finalModel               
plot(ctreeTune$finalModel)

testResults$cTree <- predict(ctreeTune, train1[, -13])

### Section 8.2 Regression Model Trees and 8.3 Rule-Based Models

set.seed(111)
m5Tune <- train(x = train1[, -13], y = train1[, 13],
                method = "M5", 
                trControl = ctrl,
                control = Weka_control(M = 10))
m5Tune
plot(m5Tune)
m5Tune$finalModel
plot(m5Tune$finalModel)
ruleFit <- M5Rules(y~., data = trainData, control = Weka_control(M = 10))
ruleFit

### Section 8.4 Bagged Trees

registerDoMC(5)
set.seed(111)
treebagTune <- train(x = train1[, -13], y = train1[, 13],
                     method = "treebag",
                     nbagg = 50,  
                     trControl = ctrl)

treebagTune

### Section 8.5 Random Forests

mtryGrid <- data.frame(mtry = floor(seq(10, ncol(train1[, -13]), length = 10)))

### Tune the model using cross-validation
set.seed(111)
rfTune <- train(x = train1[, -13], y = train1[, 13],
                method = "rf",
                tuneGrid = mtryGrid,
                ntree = 1000,
                importance = TRUE,
                trControl = ctrl)
rfTune
plot(rfTune)
rfImp <- varImp(rfTune, scale = FALSE)
rfImp

ctrlOOB <- trainControl(method = "oob")
set.seed(111)
rfTuneOOB <- train(x = train1[, -13], y = train1[, 13],
                   method = "rf",
                   tuneGrid = mtryGrid,
                   ntree = 1000,
                   importance = TRUE,
                   trControl = ctrlOOB)
rfTuneOOB
plot(rfTuneOOB)

### Tune the conditional inference forests

set.seed(111)
condrfTune <- train(x = train1[, -13], y = train1[, 13],
                    method = "cforest",
                    tuneGrid = mtryGrid,
                    controls = cforest_unbiased(ntree = 1000), 
                    trControl = ctrl)
condrfTune
plot(condrfTune)

set.seed(111)
condrfTuneOOB <- train(x = train1[, -13], y = train1[, 13],
                       method = "cforest",
                       tuneGrid = mtryGrid,
                       controls = cforest_unbiased(ntree = 1000),
                       trControl = trainControl(method = "oob"))
condrfTuneOOB

plot(condrfTuneOOB)

### Section 8.6 Boosting

gbmGrid <- expand.grid(interaction.depth = seq(1, 7, by = 2),
                       n.trees = seq(100, 1000, by = 50),
                       shrinkage = c(0.01, 0.1),
                       n.minobsinnode = 20)
set.seed(111)
gbmTune <- train(x = train1[, -13], y = train1[, 13],
                 method = "gbm",
                 tuneGrid = gbmGrid,
                 trControl = ctrl,
                 verbose = FALSE)
gbmTune
plot(gbmTune, auto.key = list(columns = 4, lines = TRUE))
gbmImp <- varImp(gbmTune, scale = FALSE)
gbmImp

### Section 8.7 Cubist

cbGrid <- expand.grid(committees = c(1:10, 20, 50, 75, 100), 
                      neighbors = c(0, 1, 5, 9))

set.seed(111)
cubistTune <- train(train1[, -13], train1[, 13],
                    "cubist",
                    tuneGrid = cbGrid,
                    trControl = ctrl)
cubistTune
plot(cubistTune, auto.key = list(columns = 4, lines = TRUE))
cbImp <- varImp(cubistTune, scale = FALSE)
cbImp

processingTime <- endTime - startTime

processingTime 
