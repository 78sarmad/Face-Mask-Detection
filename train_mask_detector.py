# import the necessary packages
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import os

# initialize the globals
# learning rate, number of epochs and batch size
INIT_LR = 1e-4
EPOCHS = 4
BS = 20

# find current file directory and build paths
dirname = os.path.dirname(__file__)
filepath = 'dataset'
DIRECTORY = os.path.join(dirname, filepath)
CATEGORIES = ["with_mask", "without_mask"]

# load images from dataset directory
print("[INFO] Loading images from dataset...")

data = []  # store data for images
labels = []  # store data for labels

# traverse category folders in dataset directory
for category in CATEGORIES:
    path = os.path.join(DIRECTORY, category)
    # loop through each image in datasets
    for img in os.listdir(path):
        img_path = os.path.join(path, img)
        # load image into 224x224 size
        image = load_img(img_path, target_size=(224, 224))
        # convert image to array and preprocess
        image = img_to_array(image)
        image = preprocess_input(image)

        # append image data and label to arrays
        data.append(image)
        labels.append(category)

# encode labels so we can use them as numpy array
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)

# set list types as numpy arrays
data = np.array(data, dtype="float32")
labels = np.array(labels)

# split data into train and test datasets
# test dataset size is 20%
(trainX, testX, trainY, testY) = train_test_split(data, labels,
                                                  test_size=0.20, stratify=labels, random_state=78)

# augment image into multiple images making slight changes
aug = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest")

# use cnn model: mobilenetv2
baseModel = MobileNetV2(weights="imagenet", include_top=False,
                        input_tensor=Input(shape=(224, 224, 3)))

# construct head of base model
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(128, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2, activation="softmax")(headModel)

# place head model on top, head is the model we train
model = Model(inputs=baseModel.input, outputs=headModel)

# set base model layers as non-trainable so they do not get updated
for layer in baseModel.layers:
    layer.trainable = False

# compile our model using ADAM optimizer and metrics set to accuracy
print("[INFO] Compiling model...")
opt = Adam(learning_rate=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt,
              metrics=["accuracy"])

# train the head model
# fit the multiple augmented images from one image
print("[INFO] Training...")
H = model.fit(
    aug.flow(trainX, trainY, batch_size=BS),
    steps_per_epoch=len(trainX) // BS,
    validation_data=(testX, testY),
    validation_steps=len(testX) // BS,
    epochs=EPOCHS)

# make predictions on the testing set
print("[INFO] Evaluating network...")
predIdxs = model.predict(testX, batch_size=BS)

# for each image in test set, find index of label with max probability
predIdxs = np.argmax(predIdxs, axis=1)

# print a classification report
print(classification_report(testY.argmax(axis=1), predIdxs,
                            target_names=lb.classes_))

# save the model to disk in h5 format
print("[INFO] Saving mask detector model...")
model.save("mask_detector.model", save_format="h5")

# plot the training loss and accuracy in plot.png file
N = EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig("plot.png")
