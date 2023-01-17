from keras.layers import Conv2D, UpSampling2D
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from skimage.color import rgb2lab, lab2rgb
from skimage.transform import resize
from skimage.io import imsave
import numpy as np
import tensorflow as tf

path = 'images/'

#Normalize images
train_datagen = ImageDataGenerator(rescale=1. / 255)

#Resize images
train = train_datagen.flow_from_directory(path, 
                                          target_size=(256, 256), 
                                          batch_size=458, 
                                          class_mode=None)

#Convert from RGB to Lab
X =[]
Y =[]
for img in train[0]:
  try:
      lab = rgb2lab(img)
      X.append(lab[:,:,0]) 
      Y.append(lab[:,:,1:] / 128) 
  except:
     print('Error converting RGB to Lab')

X = np.array(X)
Y = np.array(Y)
X = X.reshape(X.shape+(1,))


#Encoder
model = Sequential()
model.add(Conv2D(64, (3, 3), activation='relu', padding='same', strides=2, input_shape=(256, 256, 1)))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(128, (3,3), activation='relu', padding='same', strides=2))
model.add(Conv2D(256, (3,3), activation='relu', padding='same'))
model.add(Conv2D(256, (3,3), activation='relu', padding='same', strides=2))
model.add(Conv2D(512, (3,3), activation='relu', padding='same'))
model.add(Conv2D(512, (3,3), activation='relu', padding='same'))
model.add(Conv2D(256, (3,3), activation='relu', padding='same'))

#Decoder
# ReLU = Rectified Linear Unit
model.add(Conv2D(128, (3,3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(64, (3,3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(32, (3,3), activation='relu', padding='same'))
model.add(Conv2D(16, (3,3), activation='relu', padding='same'))

# tanh = The hyperbolic tangent
model.add(Conv2D(2, (3, 3), activation='tanh', padding='same'))
model.add(UpSampling2D((2, 2)))

# Adam = Adaptive moment estimation
model.compile(optimizer='adam', loss='mse' , metrics=['accuracy'])

model.summary()
model.fit(X,Y,validation_split=0.1, epochs=100, batch_size=16)

model.save('colorize_autoencoder_VGG15_10000.model')
