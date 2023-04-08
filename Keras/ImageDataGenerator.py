
KERAS IMAGE DATA GEN
-> Used to add turn and tweaks for the image for better model learning of Variations in images
from keras.preprocessing.image import ImageDataGenerator

# Create an instance of ImageDataGenerator with specified augmentation settings
datagen = ImageDataGenerator(
    rotation_range=20,  # rotate images by up to 20 degrees
    width_shift_range=0.1,  # shift images horizontally by up to 10% of the image width
    height_shift_range=0.1,  # shift images vertically by up to 10% of the image height
    shear_range=0.2,  # apply shearing transformations
    zoom_range=0.2,  # apply zooming transformations
    horizontal_flip=True,  # flip images horizontally
    fill_mode='nearest'  # fill in newly created pixels with the nearest pixel value
)

# Generate augmented images from a directory of images
train_generator = datagen.flow_from_directory(
    'train',  # directory containing the original training images
    target_size=(224, 224),  # size of the input images expected by the model
    batch_size=32,  # number of images in each batch
    class_mode='binary'  # type of classification task (e.g., binary, categorical)
)

# Use the generated augmented images for training a deep learning model
model.fit_generator(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=10
)
