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
import random

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
    error = None
    if 'username' not in session:
        if request.method == 'POST':
            if request.form['submit_button'] == 'Sign Up':
                return redirect(url_for('signup'))
            elif request.form['submit_button'] == 'Login':
                return redirect(url_for('login'))
            else:
                error = 'User Log in is required'
    return render_template('home.html', error=error)

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

        return redirect(url_for('login'))
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
            return redirect(url_for('home'))
        else:
            error = 'Invalid username or password'
            return render_template('login.html', error=error)
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('home'))

@app.route('/mri', methods=['GET', 'POST'])
def mri():
    if 'username' not in session:
        return render_template('home.html', error='User Log in is required')

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

@app.route('/cognitive', methods=['GET', 'POST'])
def cognitive():
    if 'username' not in session:
        return render_template('home.html', error='User Log in is required')

    if request.method == 'POST':
        # Get form answers
        full_name = request.form['full_name']
        year_of_birth = request.form['year_of_birth']
        favorite_color = request.form['favorite_color']
        citizenship = request.form['citizenship']

        # Generate 6 MCQ questions
        mcq_questions = [
            {"question": "What is the color of sky?", "options": ["Red", "Green", "Blue"], "answer": 3},
            {"question": "What is 2 + 2?", "options": ["3", "4", "5"], "answer": 1},
            {"question": "Which of these is not a geometric shape?", "options": ["Circle", "Square", "Triangle", "Oval"], "answer": 3},
            {"question": "What is the capital of France?", "options": ["Paris", "London", "Berlin", "Rome"], "answer": 0},
            {"question": "Which of these is not a programming language?", "options": ["Python", "Java", "C++", "Banana"], "answer": 3},
            {"question": "What is the largest planet in our solar system?", "options": ["Earth", "Mars", "Jupiter", "Saturn"], "answer": 2}
        ]

        # Shuffle the questions
        random.shuffle(mcq_questions)

        # Store answers in the database
        conn = get_db_connection()
        c = conn.cursor()
        c.execute("INSERT INTO CognitiveTest (user_id, full_name, year_of_birth, favorite_color, citizenship, date_taken, score) VALUES (?, ?, ?, ?, ?, ?, ?)", 
                  (session['user_id'], full_name, year_of_birth, favorite_color, citizenship, datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 0))
        conn.commit()
        conn.close()

        return redirect(url_for('cognitive_test', mcq_questions=str(mcq_questions)))

    return render_template('cognitive.html')

@app.route('/cognitive/test', methods=['GET', 'POST'])
def cognitive_test():
    if 'username' not in session:
        return redirect(url_for('home'))

    mcq_questions = request.args.get('mcq_questions')
    mcq_questions = eval(mcq_questions)  # Convert the string back to a list of dictionaries

    if request.method == 'POST':
        # Get user's answers
        user_answers = []
        for i in range(len(mcq_questions)):
            answer = request.form.get(f'question_{i}')
            if answer is None:
                answer = -1  # Assign a default value or handle missing answer appropriately
            user_answers.append(int(answer))

        # Verify answers and calculate score
        score = 0
        for i, question in enumerate(mcq_questions):
            if user_answers[i] == question['answer']:
                score += 3

        # Retrieve form answers from the database
        conn = get_db_connection()
        c = conn.cursor()
        c.execute("SELECT full_name, year_of_birth, favorite_color, citizenship FROM CognitiveTest WHERE user_id = ?", (session['user_id'],))
        form_answers = c.fetchone()
        conn.close()

        # Verify form answers
        form_score = 0
        if form_answers['full_name'] == request.form['full_name']:
            form_score += 3
        if form_answers['year_of_birth'] == request.form['year_of_birth']:
            form_score += 3
        if form_answers['favorite_color'] == request.form['favorite_color']:
            form_score += 3
        if form_answers['citizenship'] == request.form['citizenship']:
            form_score += 3

        total_score = score + form_score

        # Update the score in the database
        conn = get_db_connection()
        c = conn.cursor()
        c.execute("UPDATE CognitiveTest SET score = ? WHERE user_id = ?", (total_score, session['user_id']))
        conn.commit()
        conn.close()

        return redirect(url_for('cognitive_result', total_score=total_score))

    return render_template('cognitive_test.html', mcq_questions=mcq_questions, enumerate=enumerate)



@app.route('/cognitive/result/<int:total_score>')
def cognitive_result(total_score):
    if 'username' not in session:
        return redirect(url_for('home'))

    if total_score >= 26:
        result_message = "Not Alzheimer"
    elif 20 <= total_score < 26:
        result_message = "Might have Alzheimer"
    else:
        result_message = "Consult Doctor"

    return render_template('cognitive_result.html', total_score=total_score, result_message=result_message)


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
