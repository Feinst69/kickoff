import sys
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

model = load_model('model.h5')

img_path = sys.argv[1]
img = image.load_img(img_path, target_size=(28,28), color_mode='grayscale')
img_array = image.img_to_array(img)
img_array = img_array.reshape(1,28,28,1) / 255.0

pred = model.predict(img_array)
print('Predicted digit:', np.argmax(pred))
