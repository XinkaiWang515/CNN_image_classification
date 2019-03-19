import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Flatten,Dense,Dropout
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras.optimizers import Adam
from keras import optimizers

from keras.models import Model
from keras.models import load_model
from keras.utils import plot_model

from keras.preprocessing.image import ImageDataGenerator

# import tensorflow as tf
# import keras.backend.tensorflow_backend as KTF
#
# KTF.set_session(tf.Session(config=tf.ConfigProto(device_count={'gpu': 0})))


data_file = 'train.csv'
data = np.genfromtxt(data_file, delimiter=',', dtype=str)

model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(350,350,3)))
model.add(MaxPooling2D((2,2)))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2)))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2)))

model.add(Flatten())
model.add(Dropout(0.5))

model.add(Dense(512, activation='relu'))
model.add(Dense(8, activation='sigmoid'))

# # Replicates `model` on 8 GPUs.
# # This assumes that your machine has 8 available GPUs.
# parallel_model = multi_gpu_model(model, gpus=8)
# parallel_model.compile(loss='categorical_crossentropy',
#                         optimizer=Adam(),
#                         metrics=['acc'])
#
# # This `fit` call will be distributed on 8 GPUs.
# # Since the batch size is 256, each GPU will process 32 samples.
# # parallel_model.fit(x, y, epochs=20, batch_size=256)


model.compile(loss='categorical_crossentropy',
    optimizer=Adam(),
    metrics=['acc'])


train_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    'categorical_image/train',
    target_size=(350, 350),
    batch_size=32,
    class_mode='categorical')

validation_datagen = ImageDataGenerator(rescale=1./255)

val_generator = validation_datagen.flow_from_directory(
    'categorical_image/val',
    target_size=(350, 350),
    batch_size=32,
    class_mode='categorical')

history = model.fit_generator(
        train_generator,
        steps_per_epoch=500,
        epochs=40,
        validation_data=val_generator,
        validation_steps=60)

model.save_weights('first_try.h5')