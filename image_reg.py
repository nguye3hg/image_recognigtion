import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models

(training_images, training_labels),(testing_images, testing_labels) = datasets.cifar10.load_data()
#divide rgb value to range from 0 to 1
training_images, testing_images = training_images/255, testing_images/255

class_names = ["Plane", "Car", "Bird", "Cat", "Deer", "Dog", "Frog", "Horse", "Ship", "Truck"]


#code for model
# model = models.Sequential()
# #input have to be 32x32
# model.add(layers.Conv2D(32, (3, 3), activation="relu", input_shape=(32, 32, 3)))
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Conv2D(64, (3, 3), activation="relu"))
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Conv2D(64, (3, 3), activation="relu"))
# model.add(layers.Flatten())
# model.add(layers.Dense(64, activation='relu'))
# model.add(layers.Dense(10, activation="softmax"))
#
# model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
#
# model.fit(training_images, training_labels, epochs=10, validation_data=(testing_images, testing_labels))
#
# loss, accuracy = model.evaluate(testing_images,testing_labels)
#
# print(f"Loss:{loss}")
# print(f"Accuracy:{accuracy}")
#
# model.save("image_regconigtion.model")

model=models.load_model("image_regconigtion.model")
#example predict

img = cv.imread("ship.jpg")
#convert color
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

plt.imshow(img, plt.cm.binary)
plt.show()

prediction = model.predict(np.array([img])/255)
index = np.argmax(prediction)
print(f"image is {class_names[index]}")
