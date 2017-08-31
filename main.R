#DATA LOADING And Training and test set division
set.seed(12345)
data <- read.csv("training.csv",stringsAsFactors = FALSE)
index <- sample(c(0,1),nrow(data),replace = TRUE , prob = c(0.8,0.2))
data.train <- data[index==0,]
data.test <- data[index==1,]
rm(index)
rm(data)

#Processing training set
im.train <- data.train$Image
data.train$Image <- NULL
d.train <- data.train
rm(data.train)

#Processing testing set
im.test <- data.test$Image
data.test$Image <- NULL
d.test <- data.test
rm(data.test)

# A little bit more preprocessing
d.train <- 96 - d.train
d.test <- 96 - d.test
im.train <- sapply(1:length(im.train), function(x) 
      rev(as.integer(unlist(strsplit(x = im.train[x],split = " " )))))
im.train <- t(im.train)


# Plotting an image and anylising it
i = 3402
im <- matrix(data=(im.train[i,]), nrow=96, ncol=96)
image(1:96, 1:96, im, col=gray((0:255)/255))
points(d.train$nose_tip_x[i], d.train$nose_tip_y[i],col="red")
points(d.train$left_eye_center_x[i],  d.train$left_eye_center_y[i],  col="blue")
points(d.train$right_eye_center_x[i], d.train$right_eye_center_y[i], col="green")


# Histogram Stretching
im.train <- sapply(1:nrow(im.train), function(x) 
    (255*(im.train[x,] - min(im.train[x,])))/(max(im.train[x,]) - min(im.train[x,]) ))
im.train <- t(im.train)

#Saving data for further use 
save(d.test,d.train,im.train,im.test,file = 'train_and_test_data.RData')
# load('train_and_test_data.RData')


pca <- prcomp(t(im.train))
save(pca,file = 'pca_data.Rdata')
# load('pca_data.Rdata')
plot(pca,type = 'l')

# this plot will help us to decide how many principle components to use
smoothScatter(1:ncol(pca$rotation),pca$sdev,xlim = c(0,300))
eigenVectos <- pca$rotation


# plotting various eigenfaces to recognize
for(i in 1:16)
{
tempMatrix <- t(im.train) %*% eigenVectos[,i]  
im <- matrix(tempMatrix, nrow=96, ncol=96)
image(1:96, 1:96, im, col=gray((0:255)/255))
}

# new facial keypoints for our 125 image are

# there are lot of NA values in d.train first we have to replace them with avg value of
# keypoints
colmean <- colMeans(d.train,na.rm = TRUE)
for(j in 1:30)
{
    index <- is.na(d.train[,j])
    d.train[index,j] <- colmean[j]
      
}
new_keypoints <-  t(t(d.train) %*% (eigenVectos[,1:125]^2))
new.im.train <- t(im.train) %*% eigenVectos[,1:125]
# doing a histogram stretching on the new.im.train gives
new.im.train <- t(new.im.train)
new.im.train <- sapply(1:nrow(new.im.train), function(x) 
  (255*(new.im.train[x,] - min(new.im.train[x,])))/(max(new.im.train[x,]) - min(new.im.train[x,]) ))


#Saving the data for further use
save(new.im.train,new_keypoints,file = "pca_implemented_data.Rdata")

load("pca_implemented_data.Rdata")


# NOW APPLYING MEAN PATCH SEARCH ON Processed 125 images.
im.train <- t(new.im.train)
d.train <- new_keypoints
d.train <- as.data.frame(d.train)
for(i in 1:16)
{
im <- matrix(data=(im.train[i,]), nrow=96, ncol=96)
image(1:96, 1:96, im, col=gray((0:255)/255))
points(d.train$nose_tip_x[i], d.train$nose_tip_y[i],col="red")
points(d.train$left_eye_center_x[i],  d.train$left_eye_center_y[i],  col="blue")
points(d.train$right_eye_center_x[i], d.train$right_eye_center_y[i], col="green")
}



