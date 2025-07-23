# SMS Spam Detection using Machine Learning

## Aim:
To build a machine learning model that accurately classifies SMS messages as Spam or Ham (Not Spam) using Natural Language Processing (NLP) and deploy it using Streamlit for easy user interaction.

### Requirements

- Python 3.x
- Libraries:
  - pandas  
  - numpy  
  - nltk  
  - scikit-learn  
  - streamlit  
  - pickle

## Installation

Install all dependencies using:

```bash
pip install nltk streamlit scikit-learn pandas numpy
pip install streamlit
pip install nltk
```
Also, download required NLTK data:
```bash
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```
## Project Description

This project processes raw SMS messages, cleans and tokenizes the text, and applies a TF-IDF Vectorizer to convert them into numerical features. A Multinomial Naive Bayes classifier is trained on labeled data to predict whether a given message is spam or not.

The app is built using Streamlit, allowing users to:

Enter or paste any SMS text

Instantly classify it as Spam or Ham

View model predictions with simple UI

