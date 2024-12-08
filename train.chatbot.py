import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD
import random
import nltk
import json
import pickle

# Menggunakan Sastrawi untuk stemmer bahasa Indonesia
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# Inisialisasi variabel
words = []
classes = []
documents = []
ignore_letters = ['!', '?', ',', '.']

# Membaca file intents
intents_file = open('intents.json', encoding='utf-8').read()  # Menambahkan encoding untuk file JSON
intents = json.loads(intents_file)

# Memproses intents
for intent in intents['intents']:
    for pattern in intent['patterns']:
        # Tokenisasi setiap kata
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        # Menambahkan dokumen ke korpus
        documents.append((word_list, intent['tag']))
        # Menambahkan kelas ke daftar kelas
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Stem dan ubah setiap kata menjadi huruf kecil, serta hapus duplikat
words = [stemmer.stem(w.lower()) for w in words if w not in ignore_letters]  # Menggunakan stemmer Sastrawi
words = sorted(list(set(words)))

# Mengurutkan kelas
classes = sorted(list(set(classes)))

# Menyimpan data
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

# Membuat data pelatihan
training = []
output_empty = [0] * len(classes)

# Menyiapkan set pelatihan, bag of words untuk setiap kalimat
for doc in documents:
    bag = []
    pattern_words = doc[0]
    pattern_words = [stemmer.stem(word.lower()) for word in pattern_words]  # Menggunakan stemmer Sastrawi

    # Membuat array bag of words
    for word in words:
        bag.append(1) if word in pattern_words else bag.append(0)

    # Output adalah '0' untuk setiap tag dan '1' untuk tag saat ini
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1

    training.append([bag, output_row])

# Mengacak fitur dan mengubah menjadi np.array
random.shuffle(training)
training = np.array(training, dtype=object)  # Menggunakan dtype=object jika ukuran tidak konsisten

# Membuat daftar train dan test. X - pola, Y - intents
train_x = np.array(list(training[:, 0]))
train_y = np.array(list(training[:, 1]))

print("Training data created")

# Membuat model
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

# Mengompilasi model
sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Melatih dan menyimpan model
hist = model.fit(train_x, train_y, epochs=200, batch_size=5, verbose=1)
model.save('chatbot_model.h5')

print("Model created")
