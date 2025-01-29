# Text Language Detection and Translation

This project uses machine learning to detect the language of a given text and translates it to English if necessary. The model is trained using a dataset of text samples in multiple languages and a Naive Bayes classifier.

## Libraries Used

- **Pandas**: For data handling and manipulation.
- **NumPy**: For numerical operations.
- **Scikit-learn**: For machine learning models, feature extraction, and data splitting.
- **Googletrans**: For translating detected text into English.
  
## Dataset

The dataset is a collection of text samples in multiple languages. The target variable (`language`) indicates the language of the corresponding text.

## Steps

1. **Data Loading and Exploration**: The dataset is loaded from a public URL and basic exploration is performed to understand its structure.
   
2. **Feature Extraction**: The text data is converted into numerical features using **CountVectorizer** to prepare it for machine learning.

3. **Model Training**: 
   - A **Multinomial Naive Bayes** model is used to classify the text into different languages.
   - The dataset is split into a training (70%) and testing (30%) set, and the model is trained on the training data.

4. **Language Detection and Translation**:
   - The `detect_and_translate` function detects the language of the input text and translates it to English if needed using the **Google Translate API**.

## Results

- The model classifies the input text into one of several languages and, if the detected language is not English, translates it into English.
  
## Installation

To run this project, you need to install the required libraries. You can install them using pip:

```bash
pip install pandas numpy scikit-learn googletrans==4.0.0-rc1
