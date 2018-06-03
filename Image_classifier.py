from keras.applications import VGG16
from keras.applications import imagenet_utils
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
import numpy as np
import argparse
import cv2

#Capturing the image using webcam

cam = cv2.VideoCapture(0)

cv2.namedWindow("test")

img_counter = 0

while True:
    ret, frame = cam.read()
    cv2.imshow("test", frame)
    if not ret:
        break
    k = cv2.waitKey(1)

    if k%256 == 32:
        # SPACE pressed
        img_name = "opencv_image.png"
        cv2.imwrite(img_name, frame)
        print("{} written!".format(img_name))
        break

cam.release()

cv2.destroyAllWindows()
print("classifying... ")

#Using Vgg16 to classify the images and downloading the imagenets weights and labels
#Downloading weights might take time when you run it first time, as they are around 500 Mb

inputShape = (224, 224)
preprocess = imagenet_utils.preprocess_input
Network = VGG16
model = Network(weights="imagenet")

#resizing and preprocessing image for classificatiom

image = load_img(img_name, target_size=inputShape)
image = img_to_array(image)
image = np.expand_dims(image, axis=0)
image = preprocess(image)
preds = model.predict(image)
P = imagenet_utils.decode_predictions(preds)
 
#Printing out top 5 predictions on terminal and labelling image with the top prediction

for (i, (imagenetID, label, prob)) in enumerate(P[0]):
	print("{}. {}: {:.2f}%".format(i + 1, label, prob * 100))

orig = cv2.imread(img_name)
(imagenetID, label, prob) = P[0][0]
cv2.putText(orig, "Label: {}, {:.2f}%".format(label, prob * 100),
	(10, 30), cv2.FONT_HERSHEY_TRIPLEX, 0.8, (0, 0, 255), 2)
cv2.imshow("Classification", orig)
cv2.waitKey(0)	