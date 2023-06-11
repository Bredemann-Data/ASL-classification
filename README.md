# ASL-classification
This script is part of a 1-week project I carried out during a data science qualification program.
This script creates a Convolutional neural network that is trained to classify pictures of American Sign Language signs.
The CNN is programmed using the keras API, One-Hot-Encoding and scaling of the picture data was performed by sklearn.
Multiple Convolutional Neural Networks where trained using the dataset „Sign Language MNIST“ (https://www.kaggle.com/datasets/datamunge/sign-language-mnist) which contains pictures of size 28x28, that show ASL signs,referring to letters of the Latin alphabet. 
In the end, the model with the best validation performance is tested with a separate test set.
