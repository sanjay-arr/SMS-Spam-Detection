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

This Spam Message Classifier uses machine learning to detect whether a given SMS is spam or ham (non-spam). The workflow includes data cleaning, preprocessing, and Natural Language Processing (NLP) steps like tokenization, stemming, and stopword removal. TF-IDF vectorization converts text into numerical features. Multiple classifiers—MultinomialNB, BernoulliNB, and GaussianNB—were tested, with MultinomialNB performing best. The project also includes visualizations for word distribution and class balance, and generates .pkl files for easy model deployment.
The project includes:

Data cleaning and preprocessing

Text transformation using NLP

Feature extraction using TF-IDF

Model training and evaluation

Deployment-ready pickle files for the model and vectorizer

Visualization of word frequency and class distribution

