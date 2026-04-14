from tensorflow.keras.preprocessing.image import ImageDataGenerator

def create_generators(train_dir, test_dir, img_size=(224,224), batch_size=32):
    """
    Create Keras generators for training and testing data.
    Applies data augmentation to training images.
    """
    # Training data augmentation: rescale + random transforms
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,
        zoom_range=0.1,
        horizontal_flip=True
    )
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary'
    )
    # Testing data: only rescale (no augmentation)
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary',
        shuffle=False  # Important: keep order for evaluation
    )
    return train_generator, test_generator
