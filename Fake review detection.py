# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

# Step 1: Load and Prepare the Dataset
# Assume 'dataset.csv' with columns 'review_text' and 'label' (1 = fake, 0 = genuine)
data = pd.read_csv('dataset.csv')
reviews = data['review_text']
labels = data['label']

# Step 2: Text Preprocessing
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

def preprocess_text(text):
    text = re.sub(r'\W', ' ', text)  # Remove special characters
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    text = text.lower()  # Convert to lowercase
    lemmatizer = WordNetLemmatizer()
    text = ' '.join([lemmatizer.lemmatize(word) for word in text.split() if word not in stopwords.words('english')])
    return text

# Apply preprocessing
data['cleaned_reviews'] = reviews.apply(preprocess_text)

# Step 3: Feature Extraction using TF-IDF
tfidf = TfidfVectorizer(max_features=5000)
X = tfidf.fit_transform(data['cleaned_reviews']).toarray()
y = labels

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Model Training - Naive Bayes
nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)
nb_predictions = nb_model.predict(X_test)

# Step 5: Model Training - Support Vector Machine (SVM)
svm_model = SVC(kernel='linear')
svm_model.fit(X_train, y_train)
svm_predictions = svm_model.predict(X_test)

# Step 6: Model Training - Recurrent Neural Network (RNN)
# Preprocessing for RNN
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(data['cleaned_reviews'])
X_sequences = tokenizer.texts_to_sequences(data['cleaned_reviews'])
X_padded = pad_sequences(X_sequences, maxlen=100)

# Split RNN data
X_train_rnn, X_test_rnn, y_train_rnn, y_test_rnn = train_test_split(X_padded, y, test_size=0.2, random_state=42)

# Define RNN model
rnn_model = Sequential()
rnn_model.add(Embedding(input_dim=5000, output_dim=64, input_length=100))
rnn_model.add(LSTM(units=64, return_sequences=True))
rnn_model.add(Dropout(0.5))
rnn_model.add(LSTM(units=32))
rnn_model.add(Dense(units=1, activation='sigmoid'))

# Compile and train RNN
rnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
rnn_model.fit(X_train_rnn, y_train_rnn, epochs=5, batch_size=64, validation_split=0.1)

# Predictions with RNN
rnn_predictions = (rnn_model.predict(X_test_rnn) > 0.5).astype("int32").flatten()

# Step 7: Model Evaluation
def evaluate_model(predictions, y_test):
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    return accuracy, precision, recall, f1

# Evaluate Naive Bayes
nb_accuracy, nb_precision, nb_recall, nb_f1 = evaluate_model(nb_predictions, y_test)
print(f"Naive Bayes - Accuracy: {nb_accuracy}, Precision: {nb_precision}, Recall: {nb_recall}, F1-Score: {nb_f1}")

# Evaluate SVM
svm_accuracy, svm_precision, svm_recall, svm_f1 = evaluate_model(svm_predictions, y_test)
print(f"SVM - Accuracy: {svm_accuracy}, Precision: {svm_precision}, Recall: {svm_recall}, F1-Score: {svm_f1}")

# Evaluate RNN
rnn_accuracy, rnn_precision, rnn_recall, rnn_f1 = evaluate_model(rnn_predictions, y_test_rnn)
print(f"RNN - Accuracy: {rnn_accuracy}, Precision: {rnn_precision}, Recall: {rnn_recall}, F1-Score: {rnn_f1}")
