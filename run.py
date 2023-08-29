import cv2
import tensorflow as tf

# Load the pre-trained ESRGAN model
model = tf.keras.models.load_model("path_to_esrgan_model")

# Load the input image
input_image = cv2.imread("input_image.jpg")

# Resize the image to the desired upscale factor
upscale_factor = 4
h, w, _ = input_image.shape
new_h = h * upscale_factor
new_w = w * upscale_factor
input_image_resized = cv2.resize(input_image, (new_w, new_h))

# Normalize pixel values to the range [-1, 1]
input_image_resized = input_image_resized / 127.5 - 1.0

# Upscale the image using the ESRGAN model
upscaled_image = model.predict(tf.expand_dims(input_image_resized, axis=0))[0]

# Denormalize pixel values
upscaled_image = (upscaled_image + 1.0) * 127.5

# Save the upscaled image
cv2.imwrite("upscaled_image.jpg", upscaled_image)
