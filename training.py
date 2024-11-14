import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Flatten, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def create_model():
    # Load pre-trained VGG16
    base_model = VGG16(weights='imagenet',
                      include_top=False,
                      input_shape=(150, 150, 3))

    # Freeze base model layers
    for layer in base_model.layers:
        layer.trainable = False

    # Create complete model
    model = Sequential([
        base_model,
        Flatten(),  # Changed from GlobalAveragePooling2D to Flatten as per colleague's implementation
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(3, activation='softmax')  # Changed to 3 classes with softmax
    ])

    return model

def create_data_generators(train_dir, img_size=(150, 150), batch_size=32):
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=0.2
    )

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',  # Changed to categorical for multiple classes
        subset='training',
        shuffle=True
    )

    validation_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',  # Changed to categorical for multiple classes
        subset='validation',
        shuffle=False
    )

    return train_generator, validation_generator

def train_model(model, train_generator, validation_generator, epochs=50):
    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',  # Changed to categorical_crossentropy
        metrics=['accuracy']
    )

    # Callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )

    model_checkpoint = ModelCheckpoint(
        'best_model.keras',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max'
    )

    # Train model
    history = model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=epochs,
        callbacks=[early_stopping, model_checkpoint]
    )

    return history

def plot_training_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Accuracy plot
    ax1.plot(history.history['accuracy'])
    ax1.plot(history.history['val_accuracy'])
    ax1.set_title('Model Accuracy')
    ax1.set_ylabel('Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.legend(['Train', 'Validation'])

    # Loss plot
    ax2.plot(history.history['loss'])
    ax2.plot(history.history['val_loss'])
    ax2.set_title('Model Loss')
    ax2.set_ylabel('Loss')
    ax2.set_xlabel('Epoch')
    ax2.legend(['Train', 'Validation'])

    plt.tight_layout()
    plt.show()

def predict_image(model, image_path, img_size=(150, 150)):
    img = tf.keras.preprocessing.image.load_img(
        image_path,
        target_size=img_size
    )
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.

    predictions = model.predict(img_array)
    class_names = ['Cactus', 'Pothos', 'Rosa']  # Updated class names
    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])

    return predicted_class, confidence

if __name__ == "__main__":
    # Configuration
    train_dir = '/home/rodrigo/Code/Plant Thing/Plants'  # Keep your local path
    img_size = (150, 150)
    batch_size = 32
    epochs = 50

    # Create data generators
    train_generator, validation_generator = create_data_generators(
        train_dir,
        img_size=img_size,
        batch_size=batch_size
    )

    # Create and train model
    model = create_model()
    history = train_model(model, train_generator, validation_generator, epochs=epochs)

    # Visualize results
    plot_training_history(history)