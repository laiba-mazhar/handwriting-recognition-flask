#Handwriting Recognition & Emotion 
Detection Web App 
This is a Flask-based AI project that allows users to upload handwritten images and 
perform a full analysis including author identification, emotion detection, grammar 
correction, sentiment analysis, and plagiarism detection. It includes a modern dashboard UI 
for usability and RESTful API support for developers. 

#Features 
• • Author Detection: Predicts the author of handwritten text using HOG features + SVM 
classifier 
• • Emotion Analysis: Determines emotion from the text: happy, sad, neutral, etc. 
• • Sentiment Scores: Analyzes sentiment polarity using VADER and TextBlob 
• • Grammar Checker: Identifies grammar issues using LanguageTool 
• • Plagiarism Check: Detects textual similarity using OCR + TF-IDF Cosine Similarity 
• • REST API: Easy to integrate backend endpoints using Postman 
• • Interactive UI: Clean Bootstrap dashboard with tabbed visualization 

#Technologies Used 
• • Python, Flask 
• • OpenCV 
• • pytesseract (OCR) 
• • scikit-learn (SVM model) 
• • HOG (Histogram of Oriented Gradients) 
• • TextBlob & NLTK (VADER Sentiment) 
• • LanguageTool (Grammar checking) 
• • TF-IDF + Cosine Similarity (Plagiarism detection) 
• • HTML, CSS, Bootstrap 5 (Frontend) 

#Project Structure 
handwriting-recognition-flask/ 
├── app.py 
├── author_model.py 
├── templates/ 
│   └── index.html 
├── static/ 
│   └── uploads/ 
├── cache/ 
│   └── author_model.pkl 
├── training_data/ 
│   └── authors_dataset/ 
├── docs/ 
│   ├── SLIDES.pptx 
│   ├── RECORDING.mp4 
│   ├── I22-1855_Laiba_Mazhar_CS4055.docx 
│   └── i22-1855_LaibaMazhar_4055.docx 
├── requirements.txt 
├── .gitignore 
└── README.md 

#Getting Started 
Run the following commands in your terminal: 
git clone https://github.com/laiba-mazhar/handwriting-recognition-flask.git 
cd handwriting-recognition-flask 
python -m venv venv 
venv\Scripts\activate  # On Windows 
pip install -r requirements.txt 
python app.py 

#API Endpoints 
• • /upload: POST – Upload a handwriting image 
• • /train-author: POST – Upload ZIP of handwriting samples 
• • /predict-author: POST – Upload image for author prediction 
• • /plagiarism: POST – Upload 2 images for comparison 




#Documents 
Included in the `docs/` folder: 
• • Final Report (.docx) 
• Demo Video (.mp4) 
• Slides Presentation (.pptx) 
• Usage Instructions 
 ‍
 Author 
Laiba Mazhar 
Roll No: 22i-1855 
GitHub: https://github.com/laiba-mazhar 
