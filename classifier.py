import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import os

def display_prediction(img, predicted_class, confidence):
    plt.figure(figsize=(6, 6))
    plt.imshow(img)
    plt.axis('off')
    plt.title(f'Prediction: {predicted_class}\nConfidence: {confidence:.2%}')
    plt.show()

# Load the trained model
model = load_model('best_model.keras')

# Prediction of new images
print("\nPredicting new images:")
test_dir = '/home/rodrigo/Code/Plant Thing/Test'  # Keep your local test directory

for img_name in os.listdir(test_dir):
    if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
        try:
            # Load and prepare the image
            print(f"\nProcessing image: {img_name}")
            img_path = os.path.join(test_dir, img_name)
            img = image.load_img(img_path, target_size=(150, 150))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array /= 255.0

            # Make prediction
            predictions = model.predict(img_array)
            predicted_class_index = np.argmax(predictions[0])
            confidence = np.max(predictions[0])

            # Define class names
            class_names = ['Cactus', 'Pothos', 'Rosa']
            predicted_class = class_names[predicted_class_index]

            # Display image and prediction
            display_prediction(img, predicted_class, confidence)

            # Show result
            print(f"Prediction for {img_name}: {predicted_class} (Confidence: {confidence:.2%})")
            
            # Display class probabilities
            for i, (class_name, prob) in enumerate(zip(class_names, predictions[0])):
                print(f"{class_name}: {prob:.2%}")

        except Exception as e:
            print(f"Error processing {img_name}: {str(e)}")