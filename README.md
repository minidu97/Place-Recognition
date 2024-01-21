# Place-Recognition
This script employs transfer learning to create a custom image classification model. It uses VGG16 pre-trained model and K-nearest neighbors (KNN) to retrieve similar images based on the feature set from a given dataset of day and night images. It also evaluates the model's performance using metrics like precision, recall, etc.

This script uses transfer learning to build a custom image classification model and an image similarity retrieval model using VGG16 pre-trained model and K-nearest neighbors (KNN) respectively.

It first imports necessary libraries then downloads and initializes the pre-trained VGG16 model, discarding its top classification layer. It then adds a few dense layers to it to customize it according to the project's needs. After setting the custom model, it compiles using the sparse categorical crossentropy loss function, Adam optimizer and accuracy as the evaluation metric.

It defines class labels and then uses a function to load images from a directory one by one, pre-process them and extract features using the VGG16 model. It loads images from three directories containing images shot in day and night scenarios.

These images are then split into training and test sets and the custom model is trained on this data, validating on the test set. Training stops when the validation loss does not improve in three continuous epochs.

The training and validation accuracies are plotted for visual analysis. The script then reshapes the image arrays to fit the input requirements of the KNN model, trains this model on the reshaped training data and uses it to predict labels for the test data.

Based on these predictions, confusion matrix, classification report, and precision-recall curves are examined to evaluate the model.

Lastly, this script provides a functionality to retrieve similar images via the K-NN model for any input image features. First image from the test set is used as a query and a similar image from the dataset is displayed.
