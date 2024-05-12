from flask import Flask, request, render_template
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import tensorflow_addons as tfa
import io
import base64

app = Flask(__name__)

# Load the trained model
model_path = "Models/H5/alzheimer_cnn_model"

# Define custom objects including the required metric function
custom_objects = {'F1Score': tfa.metrics.F1Score(num_classes=4)}

model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)

# Define class labels
CLASSES = ['NonDemented', 'VeryMildDemented', 'MildDemented', 'ModerateDemented']

def preprocess_image(img_data, target_size=(176, 176)):
    """Preprocesses the input image for model prediction."""
    img = image.load_img(io.BytesIO(img_data), target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array / 255.0  # Rescale pixel values to [0, 1]

def predict_image_class(img_data, model):
    """Predicts the class label and confidence for the input image."""
    processed_img = preprocess_image(img_data)
    prediction = model.predict(processed_img)[0]  # Get prediction for the single input image
    predicted_class_index = np.argmax(prediction)
    predicted_class = CLASSES[predicted_class_index]
    confidence = prediction[predicted_class_index]

    return predicted_class, confidence, prediction

def plot_image_with_prediction(img_data, predicted_class, confidence, prediction_probabilities):
    """Plots the input image with the predicted class, confidence, and top predicted classes."""
    img = image.load_img(io.BytesIO(img_data), target_size=(176, 176))
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

    # Convert the plot to a base64-encoded string
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plot_base64 = base64.b64encode(buf.read()).decode('utf-8')

    return plot_base64

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get the uploaded image file
        img_file = request.files['image']
        img_data = img_file.read()

        # Predict image class and confidence
        predicted_class, confidence, prediction_probabilities = predict_image_class(img_data, model)

        # Generate the plot as a base64-encoded string
        plot_base64 = plot_image_with_prediction(img_data, predicted_class, confidence, prediction_probabilities)

        return render_template('index.html', predicted_class=predicted_class, confidence=confidence, plot_base64=plot_base64)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)