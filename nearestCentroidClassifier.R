setwd("~/Desktop/Folder/R Projects/nearestCentroidExample")

set.seed(1)

simulatedData <- data.frame(X = c(rnorm(100), rnorm(100, mean = 2)), Y = c(rnorm(100), rnorm(100, mean = -2)))

classifications <- c(rep("red", 100), rep("blue", 100))

pdf("simulatedData.pdf", width = 5, height = 5)

plot(simulatedData, col = classifications)

dev.off()

classifications <- as.factor(classifications)

nearestCentroidClassification <- function(data, classifications, unclassifiedPoint, p = c(1, 2, Inf)) {
  Levels <- levels(classifications)
  nLevels <- length(Levels)
  diffCentroids <- data.frame(X = rep(NA, nLevels), Y = rep(NA, nLevels))
  for (i in 1:nLevels) {
    subset <- data[as.character(classifications) == Levels[i], ]
    diffCentroids[i, ] <- colMeans(subset)
  }
  for (i in 1:length(unclassifiedPoint)) {
    diffCentroids[ , i] <- abs(diffCentroids[ , i] - unclassifiedPoint[i])
  }
  if (is.infinite(p)) {
    return(Levels[which.min(diffCentroids) %% nLevels, ])
  }
  else if (is.finite(p)) {
    distances <- apply(diffCentroids, 1, function(x) sum(x^p)^(1/p))
    return(Levels[which.min(distances)])
  }
  else {
    warning("No or invalid argument given for p, defaulting to l2/euclidean distances.")
    distances <- apply(diffCentroids, 1, function(x) sum(x^2)^(1/2))
    return(Levels[which.min(distances)])
  }
}

classificationOfNewPoint <- nearestCentroidClassification(simulatedData, as.factor(classifications), unclassifiedPoint = c(1, -1.2), p = 2)

Xs = seq(-2.8, 4.5, by = 0.1)
Ys = seq(-4.6, 2.2, by = 0.1)

resultMatrix <- matrix(nrow = length(Xs)*length(Ys), ncol = 2)
pixelColour <- rep(NA, length(Xs)*length(Ys))

i <- 1

for (x in Xs) {
  for (y in Ys) {
    resultMatrix[i, ] <- c(x, y)
    pixelColour[i] <- nearestCentroidClassification(simulatedData, classifications, unclassifiedPoint = c(x, y), p = 2)
    i <- i+1
  }
}

rm(i, x, y)

pdf("classificationMap.pdf", height = 5, width = 5)

plot(x = resultMatrix[ , 1], y = resultMatrix[ , 2], col = pixelColour, cex = 0.1, xlab = "X", ylab = "Y")

dev.off()

testIndex <- sample(1:200, size = 40)

testData <- simulatedData[testIndex, ]
testClassification <- classifications[testIndex]

trainingData <- simulatedData[-testIndex, ]
trainingClassification <- classifications[-testIndex]

predictedClassification <- rep(NA, 40)

for (i in 1:40) {
  predictedClassification[i] <- nearestCentroidClassification(trainingData, trainingClassification, unclassifiedPoint = unlist(testData[i, ]), p = 2)
}

proportionCorrect <- mean(testClassification == predictedClassification)

pdf("plotActualvPredicted.pdf", height = 5, width = 5)

plot(x = c(testData[ , 1], testData[ , 1]), y = c(testData[ , 2], testData[ , 2]), col = c(predictedClassification, testClassification), cex = c(rep(0.5, 40), rep(1, 40)), pch = c(rep(19, 40), rep(1, 40)), xlab = "X", ylab = "Y")

legend(x = 2.2, y = 2.1, legend = c("Predicted", "Actual"), pch = c(19, 1))

dev.off()