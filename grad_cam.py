from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from keras.preprocessing import image
import keras.backend as K
import numpy as np
import cv2
import sys

model = VGG16(weights="imagenet")
model.summary()
# # img_path = sys.argv[1]
img = image.load_img("dog.jpg", target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)
preds = model.predict(x)
print("preds ===== ",preds)
class_idx = np.argmax(preds[0])
print("class_idx ====", class_idx)
class_output = model.output[:, class_idx]
print("class_output ===", class_output)
last_conv_layer = model.get_layer("block1_conv1")

grads = K.gradients(class_output, last_conv_layer.output)[0] # calculate gradients of each feature map wrt output

pooled_grads = K.mean(grads, axis=(0, 1, 2)) #GAP of gradients

iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])
pooled_grads_value, conv_layer_output_value = iterate([x]) # gives gradients and its respective feature map
layer_filter = conv_layer_output_value.shape[2]
print(layer_filter)
for i in range(64):
    conv_layer_output_value[:, :, i] *= pooled_grads_value[i] # multiplying GAPgradient with its respective feature map
# heatmap = np.mean(conv_layer_output_value, axis=-1)

heatmap = np.sum(conv_layer_output_value, axis=-1) #summation of featuremaps to give discriminative saliency maps

heatmap = np.maximum(heatmap, 0) # Relu

heatmap /= np.max(heatmap) # normalize

img = cv2.imread("dog.jpg")
img = cv2.resize(img, (224, 224))
heatmap = cv2.resize(heatmap, (224, 224))
heatmap = np.uint8(255 * heatmap)
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
superimposed_img = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
cv2.imshow("Original", img)
cv2.imshow("GradCam", superimposed_img)
cv2.waitKey(0)
