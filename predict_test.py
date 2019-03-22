from keras.models import load_model
from keras.preprocessing import image
import numpy as np
from keras.optimizers import RMSprop
import pandas as pd
from sklearn.metrics import accuracy_score

model = load_model('emo_detect.h5')
# model.compile(loss='categorical_crossentropy',
#               optimizer=RMSprop(lr=1e-4),
#               metrics=['accuracy'])

validation_datagen = image.ImageDataGenerator(rescale=1./255)

val_generator = validation_datagen.flow_from_directory(
    'categorical_image/val',
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical')
# eva = model.evaluate_generator(generator=val_generator, steps=val_generator.n//val_generator.batch_size)
# print(eva)

test_datagen = image.ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    'categorical_image/testfolder',
    target_size=(150, 150),
    batch_size=1,
    class_mode='categorical')

test_generator.reset()
pred=model.predict_generator(test_generator,steps=test_generator.n//test_generator.batch_size)

predicted_class_indices=np.argmax(pred,axis=1)

labels = (val_generator.class_indices)
labels = dict((v,k) for k,v in labels.items())
predictions = [labels[k] for k in predicted_class_indices]

# print(type(predictions))
filenames=test_generator.filenames
label = [name.split('/')[1].split('_')[0] for name in filenames]
acc = accuracy_score(label,predictions)
print(acc)
# results=pd.DataFrame({"Filename":filenames,
#                       "Predictions":predictions})
#
# print(results)