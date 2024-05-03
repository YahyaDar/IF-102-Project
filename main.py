import pandas as pd

from nltk.stem import WordNetLemmatizer
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

from sklearn.model_selection import train_test_split

from tensorflow.keras.callbacks import EarlyStopping

data = pd.read_csv('spam.csv')

# Data Preprocessing
data['v2'] = data['v2'].str.lower()
data['v2'] = data['v2'].str.replace('[^a-zA-Z0-9]', '')

lemmatizer = WordNetLemmatizer()
data['text'] = data['text'].apply(lambda x: ' '.join([lemmatizer.lemmatize(word) for word in x.split()]))

X = data['v2']
y = data['v1']

tonkenizer = Tokenizer(num_words=5000)
tonkenizer.fit_on_texts(X)
X_sequences = tonkenizer.texts_to_sequences(X)

max_len = 100
X_padded = pad_sequences(X_sequences, maxlen=max_len, padding='post')


# Model Building
model = Sequential()

model.add(Embedding(input_dim=5000, output_dim=128, input_length=max_len))
model.add(LSTM(units=64))
model.add(Dense(units=1, activation='sigmoid'))


# Print Model Summary
model.summary()


# Model Compilation with Optimizer, Loss and Metrics
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


# Splitting the data into training and testing
X_train, X_test, y_train, y_test = train_test_split(X_padded, y, test_size=0.2, random_state=0)

# Early Stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=2)

# Model Training
model.fit(X_train, y_train, epochs=10, batch_size=64, callbacks=[early_stopping], verbose=1)

# Model Evaluation
loss, accuracy = model.evaluate(X_test, y_test)
print('Accuracy:', accuracy)
print('Loss:', loss)
