## Importing Libraries
import streamlit as st
import numpy as np
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from scipy.spatial.distance import euclidean
import io
import PyPDF2
import pytesseract
from PIL import Image
import pathlib
import docx2txt 
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import pathlib
import os
import spacy
import pdfminer
from pdfminer.converter import TextConverter
from pdfminer.pdfinterp import PDFPageInterpreter
from pdfminer.pdfinterp import PDFResourceManager
from pdfminer.pdfpage import PDFPage
from pdfminer.layout import LAParams
import ocrmypdf
import warnings
warnings.filterwarnings("ignore")

### Importing Trained Model (Trained on ------ > Kaggle Resume Dataset )

svc = pickle.load(open('svc_model','rb'))
word_vectorizer = pickle.load(open('word_vectorizer','rb'))
enc = pickle.load(open('Label_encoder','rb'))

## Importing a pre trained spacy 

# nlp = spacy.load('en_core_web_sm')
nlp = spacy.load('en_core_web_trf')

# Text Extraction Functions

def extract_text_from_pdf(uploaded_file):
    
    resource_manager = PDFResourceManager()
    output_string = io.StringIO()
    converter = TextConverter(resource_manager, output_string, laparams=LAParams())
    interpreter = PDFPageInterpreter(resource_manager, converter)
    for page in PDFPage.get_pages(uploaded_file):
        try:
            interpreter.process_page(page)
        except AttributeError:
            ocrmypdf_file = 'ocrmypdf_temp.pdf'
            ocrmypdf.ocr(page, ocrmypdf_file)
            with open(ocrmypdf_file, 'rb') as ocr_file:
                    for ocr_page in PDFPage.get_pages(ocr_file):
                        interpreter.process_page(ocr_page)
            os.remove(ocrmypdf_file)
    text = output_string.getvalue()
    text = str(text.replace('\n','\t')).replace('\t',' ')
    output_string.close()
    converter.close()
    return text


def extract_text_from_doc(doc):
    text = docx2txt.process(doc)
    text = str(text.replace('\n','\t')).replace('\t',' ')
    return text

## Category Function

def category(text):
    category = enc.inverse_transform(svc.predict(word_vectorizer.transform([text])))[0]
    return category

# Details Extration Functions

def applicant_name(docx):
    person_names = []
    for ent in docx.ents:
        if ent.label_ == 'PERSON':
            person_names.append(ent.text)
    return person_names

def phone_extract(docx):
    phone_numbers = []
    for token in docx:
        if token.like_num and len(token.text) >= 10:
            phone_numbers.append(token.text)
    if phone_numbers!=[]:
        return phone_numbers[0]
    else:
        return None

def email_extract(docx):
    emails = []
    for token in docx:
        if token.like_email:
            emails.append(token.text)
    if emails!=[]:
        return emails[0]
    else:
        return None

def skills_extract(docx):
    skills = []
    for ent in docx.ents:
        if ent.label_ == "SKILL":
            skills.append(ent.text)
            return skills    
            
def extract_experience(text):
    # doc = nlp(text)
    for ent in text.ents:
        if ent.label_ == "DATE":
            if "year" in ent.text.lower():
                return ent.text.strip()

### Resume Score Function 
def resume_score(resume,Job_desc):
    ## Co simmilarity
    
    stop_words = list(spacy.lang.en.stop_words.STOP_WORDS)
    # resume = nlp(resume)
    # Job_desc = nlp(Job_desc)
    resume_filtered = [token.text for token in resume if token.is_alpha]
    Job_desc_filtered = [token.text for token in Job_desc if token.is_alpha]
    resume_filtered =[i.lower() for i in resume_filtered]
    Job_desc_filtered = [i.lower() for i in Job_desc_filtered]
    resume_filtered= ' '.join(resume_filtered)
    Job_desc_filtered= ' '.join(Job_desc_filtered)
    vectorizer = TfidfVectorizer(sublinear_tf=True,stop_words=stop_words)
    vectorizer.fit([Job_desc_filtered])
    J_vector = vectorizer.transform([Job_desc_filtered])
    R_vector = vectorizer.transform([resume_filtered])
    similarity_score = cosine_similarity(J_vector,R_vector)[0][0]*100
    
    ## Euclidean distance
    # J_vector_array = J_vector.toarray().ravel()
    # R_vector_array = R_vector.toarray().ravel()
    # euclidean_distance = euclidean(J_vector_array, R_vector_array)
    
    ## User Category
    user_category = category(resume_filtered)
    
    ## Job vs User category score
    job_category = category(Job_desc_filtered)
    category_match_score = 1 if job_category == user_category else 0
    
    ## Skills Scores
    skills_score = len(skills_extract(resume)) if skills_extract(resume) else 0 ##Spacy en_core_web_sm not able to detect skills 
    
    ## Experience Score
    experience_score = len(extract_experience(resume)) if extract_experience(resume) else 0
    ## Give a weights to each one
    ## w1-->cosimilarity 75%
    ## w3-->job vs user category score 10%
    ## w4-->skills scores 5%, 
    ## w5-->Experience scores 10%
    
    w1=0.75
    # w2=0.20
    w3=0.10
    w4=0.05
    w5=0.05
    total_scores = (w1*similarity_score)+(w3*category_match_score)+(w4*skills_score)+(w5*experience_score)
    if (total_scores >= 60):
        result = 'PASS'
    else:
        result = 'FAIL'
    return f'RESULT:{result}', f'User Category : {user_category}.',f'Co similarity : {similarity_score}%', f"Skills score : {skills_score}%", f'Experience scores : {experience_score}%',f'Total Resume Score : {total_scores}%'


## Streamlit app 
def app():
    st.title('Resume Analyser Prototype') 
    job = st.text_input('Job Description','''machine learning, data science, pandas , numpy, sql, deep learning, computer vision , data visualisation , python ''')
       
    uploaded_file = st.file_uploader("Choose a file")
    if st.button('Upload File'):
        if uploaded_file is not None:
            file_type = uploaded_file.type
            if file_type == "application/pdf":
                resume = extract_text_from_pdf(uploaded_file)
            elif file_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                resume = extract_text_from_doc(uploaded_file)
            else:
                st.warning("Please upload a PDF or DOCX file.")
            if resume:
                resume = nlp(resume)
                job=nlp(job)
                
                st.write('Applicant Name:',applicant_name(resume))
                st.write('Contact No:',phone_extract(resume))
                st.write('Email:',email_extract(resume))
                st.write('Skills:', skills_extract(resume))
                st.write('Experience:', extract_experience(resume))
                
                st.write('Resume score:',resume_score(resume,job))
                
            
    
    
    if st.button('Click Here to Get Project info'):
        st.write('respository link',"https://github.com/Darshan85069/resume-nlp-project.git")
        im1 = Image.open("images\Prototype_Page_1.jpg")
        im2 = Image.open("images\Prototype_Page_2.jpg")
        im3 = Image.open("images\Prototype_Page_3.jpg")
        im4 = Image.open("images\Prototype_Page_4.jpg")
        im5 = Image.open("images\Prototype_Page_5.jpg")
        im6 = Image.open("images\Prototype_Page_6.jpg")
        im7 = Image.open("images\Prototype_Page_7.jpg")
        
        st.image([im1,im2,im3,im4,im5,im6,im7])
        

app()

