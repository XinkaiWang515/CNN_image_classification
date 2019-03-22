# CNN_image_classification

This project focus on building a CNN model for emotion classification from image inputs.

The dataset contains 12994 facial images including 8 emotion classes: surprise, sadness, neutral, happiness, fear, disgust, contempt and anger.

Because 12398.jpg cannot open, I delete it when splitting images into train, validation and test set. Furthermore, this is classes number: {'anger': 239, 'contempt': 8, 'disgust': 197, 'fear': 19, 'happiness': 5408, 'neutral': 6520, 'sadness': 253, 'surprise': 349}.
It's obviously imbalanced.

In training file image_gen.py, keras sequential model is used with 4 convolutional layers, maxpooling layers and 2 FC layers. Use ImageDataGenerator to augment data and train the model.
