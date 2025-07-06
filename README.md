# Resume-Analyzer-with-ML-NLP
A Streamlit web app that uses NLP and machine learning to classify resumes into job categories like Python Developer, Data Scientist, etc. It processes text using TF-IDF, predicts roles with a trained ML model, and allows users to upload resumes for instant predictions
# 🧠 NLP-Based Resume Classifier using Machine Learning

A smart resume screening web app that uses **Natural Language Processing (NLP)** and **Machine Learning (ML)** to predict the job category of a candidate based on the content of their resume. Built with Python and deployed using **Streamlit**, the app supports PDF or text resume uploads and outputs real-time predictions.

---

## 🚀 Live Demo
  Local URL: http://localhost:8501
  Network URL: http://172.18.108.19:8501

---
Resume Scoring (Out of 100)
Keyword Match (50 points):
Scores based on the percentage of job-related keywords found in the resume.
Example: If 10 out of 20 keywords match → (10/20) × 50 = 25 points.

Education Match (25 points):
Full points awarded if the resume contains any education-related keywords like "b.tech", "m.tech", or "bachelor".

Experience Match (25 points):
Full points awarded if the resume mentions work experience (e.g., "2 years", "3 yrs").

Total Score = Keyword Score + Education Points + Experience Points



## 🖼 Demo Screenshot

![Screenshot 2025-07-06 142940](https://github.com/user-attachments/assets/e0d89420-cc1a-460f-8c1e-7bf8f852fc57)




## ✨ Features

- Upload resumes (`.pdf` or `.txt`)
- Cleans and processes text with NLP
- Uses TF-IDF vectorizer for feature extraction
- Predicts job roles using a trained ML classifier
- Interactive interface built with Streamlit
- Classifies into 20+ common job categories

---

## 🛠 Tech Stack

- **Language:** Python
- **Frontend:** Streamlit
- **NLP:** NLTK, Regex
- **ML Model:** KNN Classifier (via Scikit-learn)
- **File Handling:** PyPDF2
- **Deployment:** Streamlit Cloud

---

## 📁 Project Structure
├── app.py # Streamlit Web App
├── clf.pkl # Trained ML Model
├── tfidf.pkl # TF-IDF Vectorizer
└── README.md # Project Description

📬 Contact
If you'd like to connect, feel free to reach out:

📧 Email: krishnakoushik1707@gmail.com

💼 LinkedIn:  www.linkedin.com/in/krishna-k-b262b1317
