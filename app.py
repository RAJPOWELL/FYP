from flask import Flask, render_template, request, redirect, url_for, session
import sqlite3
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import tensorflow_addons as tfa
import os
import time
from datetime import datetime


app = Flask(__name__)
app.secret_key = 'your_secret_key'

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

def get_db_connection():
    conn = sqlite3.connect('alzheimer_detection.db')
    conn.row_factory = sqlite3.Row
    return conn

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        if request.form['submit_button'] == 'Sign Up':
            return redirect(url_for('signup'))
        elif request.form['submit_button'] == 'Login':
            return redirect(url_for('login'))
    return render_template('home.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        email = request.form['email']

        conn = get_db_connection()
        c = conn.cursor()
        c.execute("INSERT INTO User (username, password, email) VALUES (?, ?, ?)", (username, password, email))
        conn.commit()
        conn.close()

        session['username'] = username
        return redirect(url_for('mri'))
    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        conn = get_db_connection()
        c = conn.cursor()
        c.execute("SELECT user_id, username FROM User WHERE username = ? AND password = ?", (username, password))
        user = c.fetchone()
        conn.close()

        if user:
            session['username'] = user['username']
            session['user_id'] = user['user_id']
            return redirect(url_for('cognitive'))
        else:
            error = 'Invalid username or password'
            return render_template('login.html', error=error)
    return render_template('login.html')


@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('home'))

@app.route('/cognitive', methods=['GET', 'POST'])
def cognitive():
    if 'username' not in session:
        return redirect(url_for('home'))

    if request.method == 'POST':
        # Get form answers
        citizenship = request.form['citizenship']
        dob = request.form['dob']
        favorite_animal = request.form['favorite_animal']
        hometown = request.form['hometown']
        favorite_color = request.form['favorite_color']

        # Store answers in the database
        conn = get_db_connection()
        c = conn.cursor()
        c.execute("INSERT INTO CognitiveTest (user_id, citizenship, dob, favorite_animal, hometown, favorite_color, date_taken, score) VALUES (?, ?, ?, ?, ?, ?, ?, ?)", (session['user_id'], citizenship, dob, favorite_animal, hometown, favorite_color, datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 0))
        conn.commit()
        conn.close()

        # Countdown
        countdown = 10
        while countdown:
            print(f"Countdown: {countdown}")
            countdown -= 1
            time.sleep(1)

        # Start test
        score = 0
        for i in range(1, 11):
            # Ask question and get answer
            question = f"Question {i}: What is 2+2?"
            answer = 4

            # Check if answer is correct
            if str(request.form.get(f'answer_{i-1}', '')) == str(answer):
                score += 3

        # Update score in the database
        conn = get_db_connection()
        c = conn.cursor()
        c.execute("UPDATE CognitiveTest SET score = ? WHERE user_id = ? AND test_id = (SELECT MAX(test_id) FROM CognitiveTest WHERE user_id = ?)", (score, session['user_id'], session['user_id']))
        conn.commit()
        conn.close()

        # Display score and graph
        return render_template('cognitive_result.html', score=score)

    return render_template('cognitive.html')


@app.route('/mri', methods=['GET', 'POST'])
def mri():
    if 'username' not in session:
        return redirect(url_for('home'))

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
    if 'username' not in session:
        return redirect(url_for('home'))

    predicted_class = request.args.get('predicted_class')
    confidence = float(request.args.get('confidence'))  # Convert to float here
    image_path = request.args.get('image_path')
    return render_template('result.html', predicted_class=predicted_class, confidence=confidence, image_path=image_path)

@app.route('/about')
def about():
    if 'username' not in session:
        return redirect(url_for('home'))
    return render_template('about.html')

if __name__ == '__main__':
    app.run(debug=True)
