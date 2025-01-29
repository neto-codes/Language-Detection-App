# Import necessary libraries
import pandas as pd  # For handling and analyzing data
import numpy as np  # For numerical operations and data manipulation
from sklearn.feature_extraction.text import CountVectorizer  # To convert text data into numerical features
from sklearn.model_selection import train_test_split  # To split the dataset into training and testing sets
from sklearn.naive_bayes import MultinomialNB  # The Naive Bayes model for classification
from googletrans import Translator  # For translating text into English

# Load the dataset from the given URL
df = pd.read_csv("https://raw.githubusercontent.com/amankharwal/Website-data/master/dataset.csv")

# Data exploration (these lines are commented out, but they can help understand the dataset)
# print(df.info())  # Displays the structure of the dataset
# print(df.head(22))  # Displays the first 10 rows of the dataset
# print(df.isnull().sum())  # Checks for missing values in the dataset
# print(df['language'].value_counts())  # Counts the occurrences of each language in the dataset

# Extract features (Text) and labels (language) from the dataset
x = np.array(df['Text'])  # Text data (features)
y = np.array(df['language'])  # Corresponding language labels

# Initialize CountVectorizer to convert text into numerical features
cv = CountVectorizer()

# Transform the text data into a numerical feature matrix
X = cv.fit_transform(x)

# Split the data into training and testing sets (70% training, 30% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize the Naive Bayes classifier and train it on the training data
model = MultinomialNB()
model.fit(X_train, y_train)

# Uncomment this line to check the model's accuracy on the test set
# print(model.score(X_test, y_test))

# Function to detect the language of a given text and translate it to English
def detect_and_translate(text):
    # Transform the input text using the trained CountVectorizer
    text_transformed = cv.transform([text])  # Converts the input into the same feature format as the model

    # Predict the language of the input text
    predicted_language = model.predict(text_transformed)[0]
    print(f"Detected Language: {predicted_language}")

    # Translate the text to English if it's not already in English
    if predicted_language != 'English':  # Only translate if the detected language is not English
        translator = Translator()  # Initialize the translator
        translated = translator.translate(text, src=predicted_language.lower(), dest='en')  # Translate text
        print(f"Translated Text: {translated.text}")  # Print the translated text
        return translated.text
    else:
        print("Text is already in English.")  # If text is English, no translation is needed
        return text

# Take user input for language detection and translation
user_input = input("Enter a text: ")
translated_text = detect_and_translate(user_input)  # Detect language and translate the input
