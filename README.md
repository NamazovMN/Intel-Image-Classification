# Intel-Image-Classification
## Idea
### The main objective of the project is classifying Intel Image dataset images into 6 classes, namely glacier, building, sea, forest, mountain and street. There are approximately 2.1 k images for each label. During the project CNN model was developed for classification task along with Transfer Learning Model VGG 16 for comparison reasons.
## Model 
Results that we will share here belong to the CNN model with 3 Convolutional Blocks (Convolutional Layer, Max Pooling Layer and Dropout Layer if it is needed) and 3 linear layers where one of them is output layer. Instead of using tensorflow's model.fit our own training session and early stopping callback were implemented. 

Additionally, dynamic model strategy that we used in CXR classification project was implemented here. By doing this, we are able to modify kernel dimensions, number of layers and other parameters easily, rather than add-delete mechanism that manual model creation demands. You can find all modifiable parameters from [utilities.py](utilities.py)
