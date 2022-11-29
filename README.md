# ML.Project.ecs171

## ABSTRACT

Our group project aims to generate a Convolutional Neural Net (CNN) that will classify .jpg input images of dogs into one of 70 breeds. Through supervised training of the model, we hope to generate a CNN that is capable of classifying a large variety of breeds to help dog owners identify the breed of their canine. 

Our data is extracted from a Kaggle dataset containing 7946 train, 700 validation, 700 test, images (of 224X224 RGB jpg format): each dog breed contains at least 78 image examples. 

The neural net will utilize a convolutional layer and hidden layer(s) to intake a .jpg image as input and output a breed class for the image (American Spaniel, Afghan, Bloodhound, etc.). 

## Preprocessing Data

As the images are of the same dimension and contain the same features of the subject and position them in the same frame, central location, and resolution, most of the logistics around centering and changing the image size are not necessary. 

Thus, our preprocessing is focused on normalizing the colored pixels for the neural net and simplifying the resolution of the images to improve the run time of the neural net. We will rescale the images so the pixel colors are associated with a normalized value, through the keras image preprocessing library. 

We will assess if data augmentation is necessary, but we are planning on initially implementing pixel normalization and resolution  reduction.

## First Model and Performing Preprocessing
We continued forward by developing a CNN model that operates on rescaled image data, where the pixels were normalized per their RGB pixel colors.

Our first model is a CNN that consists of a rescaling layer, five complexity reduction layers (MaxPooling2D- one after the rescaling and each convolutional layer), four convolutional layers, and three dense layers. As visible in our Jupyter Notebook, the training and “validation” graphs refer to the training accuracy and loss between our training set and a separate subset for testing. 

It has been labeled as validation, but it is a separate set from the training and was used to test the efficacy of the neural net. As one can tell from the graphs, since our model is increasing in complexity, but our accuracy is about 50% (random chance equivalent), and our loss is significantly higher than our training loss: the information suggests that our model may be currently overfit to our training data. We will adjust for this by creating another model- and potentially increasing more preprocessing in regards to decreasing the resolutions so we can improve speed and increase the layers in the model.
