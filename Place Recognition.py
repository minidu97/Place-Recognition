#Import necessary Python slibraries
import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_curve
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import label_binarize

#Download and initialize the pre-trained VGG16 model without top classification layer
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

#Add few dense layers to customize to fit to the project
x = base_model.output
x = Flatten()(x)#Convert features to one dimensional vector
x = Dense(64, activation='relu')(x)
predictions = Dense(3, activation='softmax')(x)

#Define the actual custom model 
model = Model(inputs=base_model.input, outputs=predictions)

#Compiling with  loss function, optimizer and evaluation metric
model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])

#Print the model architecture
print(model.summary())  

#Defining class labels and corresponding numerical indices
class_labels = {'day_left': 0, 'day_right': 1, 'night_right': 2}

#Function to load images from a directory, extracting features using the pre-trained model and return
def load_images_and_extract_features(directory):
    all_images = []
    all_features = []
    all_labels = []
    for filename in os.listdir(directory):
        img = load_img(os.path.join(directory, filename), target_size=(224, 224))
        img_data = preprocess_input(np.expand_dims(img_to_array(img), axis=0))
        all_images.append(img_data)
        all_features.append(np.squeeze(base_model.predict(img_data)))
        all_labels.append(class_labels[os.path.basename(directory)])
    return np.array(all_images), np.array(all_features), os.listdir(directory), all_labels

#Source filese 
#add the source here 
image_directories = []

#Load, extract features, and store labels 
all_images = []
all_features = []
all_filenames = []
all_labels = []
for dir in image_directories:
    dir_images, dir_features, dir_filenames, dir_labels = load_images_and_extract_features(dir)
    all_images.extend(dir_images)
    all_features.extend(dir_features)
    all_filenames.extend(dir_filenames)
    all_labels.extend(dir_labels)

#Further processing convert the lists to numpy array
all_images = np.array(all_images).reshape(-1, 224, 224, 3)
all_features = np.array(all_features).reshape(-1, 7, 7, 512)
all_labels = np.array(all_labels)

print(f"Shape of all_features: {all_features.shape}")
print(f"Shape of all_images: {all_images.shape}")

#Split the dataset into a training and test
X_train, X_test, y_train, y_test = train_test_split(all_images, all_labels, test_size=0.2, random_state=42)

#Callback for early stopping of training, when the validation loss does not improve for 3 consecutive epochs
early_stopping = EarlyStopping(monitor='val_loss', patience=3)

#Train the model using training set and validting by test set
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, callbacks=[early_stopping])

#Plot training accuracy and validation accuracy
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

#Reshape X_train and X_test data where each row corresponds to an image and column corresponds to a pixel in the flattened image
#because changing the dimensions of the image arrays to 2D arrays where each row corresponds to an image
#do this to match the input requirements of the KNN model
X_train_reshaped = X_train.reshape((X_train.shape[0], -1))
X_test_reshaped = X_test.reshape((X_test.shape[0], -1))

#Initialize and fit a 1-NN model on the reshaped training data
nn_model = NearestNeighbors(n_neighbors=1, metric='euclidean').fit(X_train_reshaped)

#Predict the labels of the test data
y_pred = model.predict(X_test)

#Convert label probabilities to actual class predictions
y_pred_class = np.argmax(y_pred, axis=1)

#Calculate the confusion matrix of our predictions
confusion_mtx = confusion_matrix(y_test, y_pred_class)
print('Confusion Matrix:')
print(confusion_mtx)

#Print a classification report
print('Classification Report:')
target_names = ['day_left', 'day_right', 'night_right'] # These should match the names of your classes
print(classification_report(y_test, y_pred_class, target_names=target_names))

#Set up a dictionary for precision and recall values
precision = dict()
recall = dict()

#Number of classes in the dataset
n_classes = 3

#Binarize the test labels and predicted labels
y_test_class_bin = label_binarize(y_test, classes=[0, 1, 2])
y_pred_class_bin = label_binarize(y_pred_class, classes=[0, 1, 2])

#Calculate precision-recall curve
for i in range(n_classes):
    precision[i], recall[i], _ = precision_recall_curve(y_test_class_bin[:, i], y_pred_class_bin[:, i])
    plt.plot(recall[i], precision[i], lw=2, label='class {}'.format(i))

#Show the precision-recall plot
plt.xlabel("recall")
plt.ylabel("precision")
plt.legend(loc="best")
plt.title("precision vs. recall curve")
plt.show()

#Function to retrieve a similar image from the KNN model
def retrieve_similar_image(query_features):
    query_features_reshaped = query_features.reshape(1, -1)
    distance, indices = nn_model.kneighbors(query_features_reshaped)
    #add the source here
    folder_paths = []
    for path in folder_paths:
        full_path = os.path.join(path, all_filenames[indices[0][0]])
        if os.path.exists(full_path):
            return full_path

    return "File not found"

#Get the features of a query image - first image in the test set
query_features = X_test_reshaped[0]

#Retrieve the filepath of a similar image
similar_image_filepath = retrieve_similar_image(query_features)

#Load and display the similar image
from PIL import Image
similar_image = Image.open(similar_image_filepath)
similar_image.show()