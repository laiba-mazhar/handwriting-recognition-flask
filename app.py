import os
import cv2
import numpy as np
import pytesseract
from flask import Flask, render_template, request, jsonify
import matplotlib.pyplot as plt
import io
import base64
from textblob import TextBlob
import language_tool_python
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from PIL import Image
from werkzeug.utils import secure_filename
import zipfile
import shutil
from author_model import train_author_model, predict_author

# Download required NLTK data
nltk.download('vader_lexicon')

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['TRAIN_FOLDER'] = 'training_data'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['TRAIN_FOLDER'], exist_ok=True)

# Initialize tools
language_tool = language_tool_python.LanguageTool('en-US')
sentiment_analyzer = SentimentIntensityAnalyzer()
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Update if needed


def preprocess_image_for_handwriting(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    inv = cv2.bitwise_not(gray)
    norm = cv2.normalize(inv, None, 0, 255, cv2.NORM_MINMAX)
    blur = cv2.GaussianBlur(norm, (5, 5), 0)
    _, threshed = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((2, 2), np.uint8)
    morph = cv2.morphologyEx(threshed, cv2.MORPH_CLOSE, kernel)
    return morph


def extract_text_from_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Unable to load image from {image_path}")
    processed = preprocess_image_for_handwriting(image)
    config = r'--oem 1 --psm 11'
    text = pytesseract.image_to_string(processed, config=config)
    return text, processed


def analyze_text(text):
    vader_scores = sentiment_analyzer.polarity_scores(text)
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity

    emotion = "neutral"
    if polarity > 0.5:
        emotion = "very positive"
    elif polarity > 0.1:
        emotion = "positive"
    elif polarity < -0.5:
        emotion = "very negative"
    elif polarity < -0.1:
        emotion = "negative"

    grammar_issues = []
    for error in language_tool.check(text):
        grammar_issues.append({
            "message": error.message,
            "replacements": error.replacements[:3],
            "context": error.context,
            "offset": error.offset,
            "length": error.errorLength
        })

    return {
        "vader_scores": vader_scores,
        "polarity": polarity,
        "subjectivity": subjectivity,
        "emotion": emotion,
        "grammar_errors": grammar_issues
    }


def create_sentiment_plot(scores):
    labels = ['Negative', 'Neutral', 'Positive', 'Compound']
    values = [scores['neg'], scores['neu'], scores['pos'], scores['compound']]
    colors = ['red', 'gray', 'green', 'blue']

    plt.figure(figsize=(8, 5))
    bars = plt.bar(labels, values, color=colors)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f'{yval:.2f}', ha='center')
    plt.ylim(0, 1.1)
    plt.title("Sentiment Analysis")
    plt.ylabel("Score")

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode('utf-8')
    return encoded


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in request'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        try:
            file.save(filepath)
            text, processed = extract_text_from_image(filepath)
            analysis = analyze_text(text)
            viz = create_sentiment_plot(analysis['vader_scores'])

            processed_filename = 'processed_' + filename
            processed_path = os.path.join(app.config['UPLOAD_FOLDER'], processed_filename)
            cv2.imwrite(processed_path, processed)

            return jsonify({
                'success': True,
                'extracted_text': text,
                'sentiment': {
                    "emotion": analysis['emotion'],
                    "vader_scores": analysis['vader_scores'],
                    "polarity": analysis['polarity'],
                    "subjectivity": analysis['subjectivity']
                },
                'grammar_errors': analysis['grammar_errors'],
                'sentiment_visualization': viz,
                'original_image': filename,
                'processed_image': processed_filename
            }), 200

        except Exception as e:
            print(f"[Upload Error] {str(e)}")
            return jsonify({'error': str(e)}), 500

    return jsonify({'error': 'Unknown error occurred'}), 500


@app.route('/plagiarism', methods=['POST'])
def plagiarism_check():
    try:
        file1 = request.files.get('file1')
        file2 = request.files.get('file2')

        if not file1 or not file2:
            return jsonify({'error': 'Both files must be uploaded'}), 400

        text1 = pytesseract.image_to_string(Image.open(file1))
        text2 = pytesseract.image_to_string(Image.open(file2))

        vectorizer = TfidfVectorizer().fit_transform([text1, text2])
        similarity = cosine_similarity(vectorizer[0:1], vectorizer[1:2])[0][0] * 100

        return jsonify({
            'text1': text1,
            'text2': text2,
            'similarity': round(similarity, 2)
        })

    except Exception as e:
        print(f"[Plagiarism Check Error] {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/train-author', methods=['POST'])
def train_author():
    zip_file = request.files.get('zipfile')
    if not zip_file:
        print("[DEBUG] No zip file uploaded.")
        return jsonify({'error': 'No zip file uploaded'}), 400

    zip_path = os.path.join(app.config['UPLOAD_FOLDER'], 'authors.zip')
    zip_file.save(zip_path)
    print(f"[DEBUG] Saved zip to {zip_path}")

    extract_path = app.config['TRAIN_FOLDER']
    if os.path.exists(extract_path):
        shutil.rmtree(extract_path)
    os.makedirs(extract_path, exist_ok=True)

    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
        print(f"[DEBUG] Extracted zip to {extract_path}")
    except Exception as e:
        print(f"[ERROR] Failed to extract zip: {e}")
        return jsonify({'error': f"Failed to extract zip: {e}"}), 500

    try:
        from author_model import train_author_model
        train_author_model(extract_path)
        print("[DEBUG] Training finished successfully.")
        return jsonify({'message': 'Model trained successfully!'}), 200
    except Exception as e:
        print(f"[ERROR] Training failed: {e}")
        return jsonify({'error': f"Training failed: {e}"}), 500



@app.route('/predict-author', methods=['POST'])
def predict_author_route():
    file = request.files.get('file')
    if not file:
        return jsonify({'error': 'No image file provided'}), 400

    try:
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
        file.save(image_path)

        author, confidence = predict_author(image_path)

        return jsonify({
            'predicted_author': author,
            'confidence': confidence,
            'uploaded_image': os.path.basename(image_path)
        }), 200

    except Exception as e:
        print(f"[Prediction Error] {str(e)}")
        return jsonify({'error': f"Prediction failed: {str(e)}"}), 500




if __name__ == '__main__':
    app.run(debug=True)
