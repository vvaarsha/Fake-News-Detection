# Corrected Code

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import pickle
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os

# Preprocess text (a more robust preprocessing step)
def preprocess_text(text):
    return text.lower().strip()

# Function to load, combine, and train the model
def train_model():
    # Load datasets
    fake_df = pd.read_csv('data/Fake.csv')
    real_df = pd.read_csv('data/True.csv')

    fake_df['label'] = 'FAKE'
    real_df['label'] = 'REAL'

    df = pd.concat([fake_df, real_df], ignore_index=True)

    # Preprocess text
    df['text'] = df['text'].apply(preprocess_text)

    print("Label Distribution in Dataset:")
    print(df['label'].value_counts())

    # Balance the dataset
    df_fake = df[df['label'] == 'FAKE']
    df_real = df[df['label'] == 'REAL']
    df_balanced = pd.concat([df_fake.sample(len(df_real), random_state=42), df_real])

    X = df_balanced['text']
    y = df_balanced['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    model = RandomForestClassifier(n_estimators=100, max_depth=30, random_state=42)
    model.fit(X_train_tfidf, y_train)

    y_pred = model.predict(X_test_tfidf)
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # Save model and vectorizer
    os.makedirs('models', exist_ok=True)
    with open('models/model.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open('models/vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)

    print("Model and vectorizer saved!")

# Flask app setup
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load the model and vectorizer from the saved files
model = pickle.load(open('models/model.pkl', 'rb'))
vectorizer = pickle.load(open('models/vectorizer.pkl', 'rb'))

# Define allowed file extensions for CSV uploads
ALLOWED_EXTENSIONS = {'csv'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/check_authenticity', methods=['POST'])
def check_authenticity():
    # Handle text input
    content = request.form['content']
    if content.strip():
        # Preprocess the input text
        preprocessed_content = preprocess_text(content)
        
        # Vectorize the input using the same vectorizer from training
        content_vectorized = vectorizer.transform([preprocessed_content])
        
        # Make a prediction
        prediction = model.predict(content_vectorized)
        
        # Map the prediction to a human-readable format
        result = "The news is REAL." if prediction[0] == 'REAL' else "The news is FAKE."
        
        # Render the result on the page
        return render_template('index.html', result=result, article_content=content)

    # Handle CSV file upload
    if 'file' not in request.files:
        return render_template('index.html', result="No file uploaded.", article_content=None)

    file = request.files['file']
    if file.filename == '':
        return render_template('index.html', result="No file selected.", article_content=None)

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Load CSV and make predictions for each article
        df = pd.read_csv(file_path)
        if 'text' not in df.columns:
            return render_template('index.html', result="CSV file must have a 'text' column.", article_content=None)

        # Preprocess and make predictions
        df['text'] = df['text'].apply(preprocess_text)
        df['prediction'] = model.predict(vectorizer.transform(df['text']))
        df['prediction'] = df['prediction'].apply(lambda x: 'The news is REAL.' if x == 'REAL' else 'The news is FAKE.')

        predictions = df[['text', 'prediction']].to_html(classes='table table-bordered')

        return render_template('index.html', result="CSV processed successfully.", predictions=predictions)

    return render_template('index.html', result="Invalid file format. Please upload a CSV file.", article_content=None)

if __name__ == '__main__':
    app.run(debug=True)

# HTML file: index.html
html_code = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>News Authenticity Checker</title>
</head>
<body>
    <h1>News Authenticity Checker</h1>
    
    <!-- Form for text input and file upload -->
    <form action="/check_authenticity" method="POST" enctype="multipart/form-data">
        <textarea name="content" rows="10" cols="50" placeholder="Enter news content here..."></textarea>
        <br><br>

        <!-- File upload field -->
        <label for="file">Or upload a CSV file:</label>
        <input type="file" name="file" id="file" accept=".csv">
        <br><br>

        <button type="submit">Check Authenticity</button>
    </form>
    
    {% if result %}
        <h2>Result: {{ result }}</h2>
        {% if article_content %}
            <p>Article Content: {{ article_content }}</p>
        {% endif %}
        {% if predictions %}
            <h3>Prediction Results from CSV:</h3>
            {{ predictions|safe }}
        {% endif %}
    {% endif %}
</body>
</html>
'''

# Save HTML file
with open('templates/index.html', 'w') as f:
    f.write(html_code)

# To train the model
if __name__ == "__main__":
    train_model()
