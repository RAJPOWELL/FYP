from flask import Flask, render_template, request, redirect, url_for
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import tensorflow_addons as tfa
import os

app = Flask(__name__)

# Load the trained model
model_path = "Models/H5/alzheimer_cnn_model"

# Define custom objects including the required metric function
custom_objects = {'F1Score': tfa.metrics.F1Score(num_classes=4)}

model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)

# Define class labels
CLASSES = ['NonDemented', 'VeryMildDemented', 'MildDemented', 'ModerateDemented']

def preprocess_image(image_path, target_size=(176, 176)):
    """Preprocesses the input image for model prediction."""
    img = image.load_img(image_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array / 255.0  # Rescale pixel values to [0, 1]

def predict_image_class(image_path, model):
    """Predicts the class label and confidence for the input image."""
    processed_img = preprocess_image(image_path)
    prediction = model.predict(processed_img)[0]  # Get prediction for the single input image
    predicted_class_index = np.argmax(prediction)
    predicted_class = CLASSES[predicted_class_index]
    confidence = prediction[predicted_class_index]

    return predicted_class, confidence, prediction

def plot_image_with_prediction(image_path, predicted_class, confidence, prediction_probabilities):
    """Plots the input image with the predicted class, confidence, and top predicted classes."""
    img = image.load_img(image_path, target_size=(176, 176))
    plt.figure(figsize=(12, 6))

    # Plot the image
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.axis('off')
    plt.title(f'Predicted Class: {predicted_class}\nConfidence: {confidence*100:.2f}%')

    # Sort predicted probabilities and get top classes
    sorted_indices = np.argsort(prediction_probabilities)[::-1]
    top_classes = [CLASSES[i] for i in sorted_indices[:3]]  # Get top 3 classes
    top_probabilities = prediction_probabilities[sorted_indices][:3]  # Corresponding probabilities

    # Plot top predicted classes and probabilities
    plt.subplot(1, 2, 2)
    y_pos = np.arange(len(top_classes))
    plt.barh(y_pos, top_probabilities, align='center', alpha=0.5)
    plt.yticks(y_pos, top_classes)
    plt.xlabel('Probability')
    plt.title('Top Predicted Classes')

    plt.tight_layout()
    plt.savefig('static/prediction_plot.png')

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/mri', methods=['GET', 'POST'])
def mri():
    if request.method == 'POST':
        image_file = request.files['image']
        image_path = os.path.join('static', image_file.filename)
        image_file.save(image_path)

        # Predict image class and confidence
        predicted_class, confidence, prediction_probabilities = predict_image_class(image_path, model)

        # Display prediction with top predicted classes
        plot_image_with_prediction(image_path, predicted_class, confidence, prediction_probabilities)

        return redirect(url_for('result', predicted_class=predicted_class, confidence=confidence*100, image_path='prediction_plot.png'))

    return render_template('mri.html')

@app.route('/result')
def result():
    predicted_class = request.args.get('predicted_class')
    confidence = float(request.args.get('confidence'))  # Convert to float here
    image_path = request.args.get('image_path')
    return render_template('result.html', predicted_class=predicted_class, confidence=confidence, image_path=image_path)


@app.route('/cognitive')
def cognitive():
    return render_template('cognitive.html')

@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == '__main__':
    app.run(debug=True)
