import streamlit as st
import pickle
import re
import nltk
import joblib
import sys
import PyPDF2
from sklearn.exceptions import ConvergenceWarning
from nltk.corpus import stopwords

# Setup
sys.modules['sklearn.preprocessing.label'] = sys.modules['sklearn.preprocessing._label']



# Load models
clf = pickle.load(open('clf.pkl', 'rb'))
tfidfd = pickle.load(open('tfidf.pkl', 'rb'))

# Category ID to name
category_mapping = {
    15: "Java Developer", 23: "Testing", 8: "DevOps Engineer", 20: "Python Developer",
    24: "Web Designing", 12: "HR", 13: "Hadoop", 3: "Blockchain", 10: "ETL Developer",
    18: "Operations Manager", 6: "Data Science", 22: "Sales", 16: "Mechanical Engineer",
    1: "Arts", 7: "Database", 11: "Electrical Engineering", 14: "Health and fitness",
    19: "PMO", 4: "Business Analyst", 9: "DotNet Developer", 2: "Automation Testing",
    17: "Network Security Engineer", 21: "SAP Developer", 5: "Civil Engineer", 0: "Advocate",25: "AI Engineer"
}

# Resume cleaning
def clean_resume(resume_text):
    clean_text = re.sub('http\S+\s', ' ', resume_text)
    clean_text = re.sub('@\S+', ' ', clean_text)
    clean_text = re.sub('#\S+', ' ', clean_text)
    clean_text = re.sub('\s+', ' ', clean_text)
    clean_text = re.sub('RT|cc', ' ', clean_text)
    clean_text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', clean_text)
    clean_text = re.sub(r'[^\x00-\x7f]', ' ', clean_text)
    return clean_text

# Main App
def main():
    nltk.download('punkt')
    nltk.download('stopwords')
    st.set_page_config(page_title="Resume Screening App", page_icon="📄")
    st.title("📄 Resume Screening App")
    st.markdown("Upload a resume to predict the job category using a machine learning model.")
    st.markdown("---")

    st.sidebar.title("🔎 About")
    st.sidebar.info("This app was built using Streamlit and ML to classify resumes.")
    st.sidebar.markdown("Created by Krishna Koushik")
    st.sidebar.markdown("---")

    upload_file = st.file_uploader('📁 Upload Resume (.pdf or .txt)', type=['pdf', 'txt'])

    if upload_file is not None:
        try:
            if upload_file.name.endswith(".txt"):
                resume_text = upload_file.read().decode("utf-8")
            else:
                pdf_reader = PyPDF2.PdfReader(upload_file)
                resume_text = ""
                for page in pdf_reader.pages:
                    resume_text += page.extract_text()
        except Exception as e:
            st.error(f"❌ Error reading file: {e}")
            return

        if resume_text.strip():
            with st.spinner("Analyzing resume..."):
                cleaned_resume = clean_resume(resume_text)
                input_features = tfidfd.transform([cleaned_resume])
                prediction_id = clf.predict(input_features)[0]
                category_name = category_mapping.get(prediction_id, "Unknown")
                st.success(f"🎯 **Predicted Category:** {category_name}")
        else:
            st.warning("⚠️ Could not extract text from the file.")

    st.markdown("---")
    st.markdown("📌 *Model uses NLP and ML for text classification*")


# Python main
if __name__ == "__main__":
    main()
