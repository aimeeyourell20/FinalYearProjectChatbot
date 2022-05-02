import json
import pickle
import random

import nltk
import numpy
from nltk.stem import LancasterStemmer
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.models import model_from_yaml

#This allows us to load our JSON data
nltk.download('punkt')

stemmer = LancasterStemmer()

with open("intents.json") as file:
    data = json.load(file)

try:
    with open("chatbot.pickle", "rb") as file:
        words, l, train, o = pickle.load(file)

#Extracting our data
#It takes the data we want from the JSON file
except:
    words = []
    l = []
    dx = []
    dy = []

# Loops through our JSON data and it extracts the data which is wanted
    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            wrds = nltk.word_tokenize(pattern)
            words.extend(wrds)
            dx.append(wrds)
            dy.append(intent["tag"])

        if intent["tag"] not in l:
            l.append(intent["tag"])

# Word stemming
# Helps to retrieve the root of the word
    words = [stemmer.stem(w.lower()) for w in words if w != "?"]
    words = sorted(list(set(words)))

    l = sorted(l)

# We have loaded the data and created the word stemming
# Represents each sentence with a list the length of the amount of words in the vocab
# Formats our outputs which will make sense to the neural network
    train = []
    o = []

    output_empty = [0 for _ in range(len(l))]

    for x, doc in enumerate(dx):
        bag = []

        wrds = [stemmer.stem(w.lower()) for w in doc]

        for w in words:
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)

        output_row = output_empty[:]
        output_row[l.index(dy[x])] = 1

        train.append(bag)
        o.append(output_row)

# Convert the training data and output to numpy arrays
    train = numpy.array(train)
    o = numpy.array(o)

    with open("chatbot.pickle", "wb") as file:
        pickle.dump((words, l, train, o), file)
#Training and saving the model
try:
    yaml_file = open('chatbotmodel.yaml', 'r')
    loaded_model_yaml = yaml_file.read()
    yaml_file.close()
    myChatModel = model_from_yaml(loaded_model_yaml)
    myChatModel.load_weights("chatbotmodel.h5")
    print("Loaded model from disk")

except:
    # Make our neural network
    myChatModel = Sequential()
    myChatModel.add(Dense(8, input_shape=[len(words)], activation='relu'))
    myChatModel.add(Dense(len(l), activation='softmax'))

    # optimize the model
    myChatModel.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # train the model
    myChatModel.fit(train, o, epochs=1000, batch_size=8)

    # serialize model to yaml and save it to disk
    model_yaml = myChatModel.to_json()
    with open("chatbotmodel.yaml", "w") as y_file:
        y_file.write(model_yaml)

    # serialize weights to HDF5
    myChatModel.save_weights("chatbotmodel.h5")
    print("Saved model from disk")


#Gets input from user and converts it to a bag of words
#Gets a prediction
#Picks an apropriate response
def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1

    return numpy.array(bag)

def chatWithBot(inputText):
    currentText = bag_of_words(inputText, words)
    currentTextArray = [currentText]
    numpyCurrentText = numpy.array(currentTextArray)

    result = myChatModel.predict(numpyCurrentText[0:1])
    result_index = numpy.argmax(result)
    tag = l[result_index]

    if result[0][result_index] > 0.7:
        for tg in data["intents"]:
            if tg['tag'] == tag:
                responses = tg['responses']

        return random.choice(responses)

    else:
        return "Sorry I do not understand, please try again"


def chat():
    while True:
        user = input("You: ")
        if user.lower() == "quit":
            break

        print(chatWithBot(user))