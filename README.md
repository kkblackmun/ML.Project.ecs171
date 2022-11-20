# ML.Project.ecs171

## ABSTRACT

Our group project aims to generate a Convolutional Neural Net (CNN) that will classify .jpg input images of dogs into one of 70 breeds. Through supervised training of the model, we hope to generate a CNN that is capable of classifying a large variety of breeds to help dog owners identify the breed of their canine. 

Our data is extracted from a Kaggle dataset containing 7946 train, 700 validation, 700 test, images (of 224X224 RGB jpg format): each dog breed contains at least 78 image examples. 

The neural net will utilize a convolutional layer and hidden layer(s) to intake a .jpg image as input and output a breed class for the image (American Spaniel, Afghan, Bloodhound, etc.). 

## Preprocessing Data

As the images are of the same dimension and contain the same features of the subject and position them in the same frame, central location, and resolution, most of the logistics around centering and changing the image size are not necessary. 

Thus, our preprocessing is focused on normalizing the colored pixels for the neural net and simplifying the resolution of the images to improve the run time of the neural net. We will rescale the images so the pixel colors are associated with a normalized value, through the keras image preprocessing library. 

We will assess if data augmentation is necessary, but we are planning on initially implementing pixel normalization and resolution  reduction.
