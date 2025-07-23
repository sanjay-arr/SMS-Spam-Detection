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

✅ **The project includes:**

- 📥 **Data cleaning and preprocessing** – Removing duplicates, null values, and irrelevant columns from the dataset.  
- 🧹 **Text normalization** – Lowercasing, removing punctuation, and filtering non-alphanumeric characters.  
- 🧠 **Text transformation using NLP** – Applying tokenization, stopword removal, stemming (using PorterStemmer).  
- 🔠 **Feature extraction using TF-IDF** – Converting text into numerical format using TF-IDF Vectorizer with 3000 max features.  
- 🧪 **Model training and evaluation** – Comparing GaussianNB, MultinomialNB, and BernoulliNB classifiers using accuracy, precision, and confusion matrix.  
- 📊 **Visualization of word frequency and class distribution** – WordClouds and bar graphs to show common words in spam and ham messages.  
- 🧾 **Corpus creation for spam/ham analysis** – Extracting and analyzing most frequent words separately in spam and ham messages.  
- 🔄 **Train-test split** – Using scikit-learn to divide the dataset for effective model validation.  
- 💾 **Deployment-ready pickle files** – Saving trained model and TF-IDF vectorizer using `pickle` for easy reusability.  
- 🧪 **Evaluation metrics** – Including accuracy, confusion matrix, and precision to assess model performance.  
- 🖼️ **Pie chart visualization** – To show distribution of spam and ham messages.  
- 📂 **Modular code structure** – Organized Jupyter notebook and Python script (`app.py`) for training and deployment.  
- 🌐 **Streamlit app integration** – Optional UI built with Streamlit to run in the browser locally.  
- 🔍 **Label encoding** – Mapping 'ham' and 'spam' text labels to numerical values using `LabelEncoder`.  
- 📦 **External libraries used** – NLTK, scikit-learn, matplotlib, seaborn, pickle, and pandas.


