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
        w, l, t, o = pickle.load(file)

#Extracting our data
#It takes the data we want from the JSON file
except:
    w = []
    l = []
    dx = []
    dy = []

#Loops through our JSON data and it extracts the data which is wanted
    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            wrds = nltk.word_tokenize(pattern)
            w.extend(wrds)
            dx.append(wrds)
            dy.append(intent["tag"])

        if intent["tag"] not in l:
            l.append(intent["tag"])

#Word stemming
#Helps to retrieve the root of the word
    w = [stemmer.stem(w.lower()) for w in w if w != "?"]
    w = sorted(list(set(w)))

    l = sorted(l)

#We have loaded the data and created the word stemming
#Represents each sentence with a list the length of the amount of words in the vocab
#Formats our outputs which will make sense to the neural network
    t = []
    o = []

    oempty = [0 for _ in range(len(l))]

    for x, doc in enumerate(dx):
        bag = []

        wrds = [stemmer.stem(w.lower()) for w in doc]

        for w in w:
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)

        orow = oempty[:]
        orow[l.index(dy[x])] = 1

        t.append(bag)
        o.append(orow)

#Convert the training data and output to numpy arrays
    t = numpy.array(t)
    o = numpy.array(o)

    with open("chatbot.pickle", "wb") as file:
        pickle.dump((w, l, t, o), file)

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
    myChatModel.add(Dense(8, input_shape=[len(w)], activation='relu'))
    myChatModel.add(Dense(len(l), activation='softmax'))

    # optimize the model
    myChatModel.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # train the model
    myChatModel.fit(t, o, epochs=1000, batch_size=8)

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
def bag_of_words(s, w):
    bag = [0 for _ in range(len(w))]

    sw = nltk.word_tokenize(s)
    sw = [stemmer.stem(word.lower()) for word in sw]

    for se in sw:
        for i, w in enumerate(w):
            if w == se:
                bag[i] = 1

    return numpy.array(bag)

def chatWithBot(inputText):
    t = bag_of_words(inputText, w)
    tl = [t]
    numpyText = numpy.array(tl)

    re = myChatModel.predict(numpyText[0:1])
    reindex = numpy.argmax(re)
    tag = l[reindex]

    if re[0][reindex] > 0.7:
        for tg in data["intents"]:
            if tg['tag'] == tag:
                respon = tg['responses']

        return random.choice(respon)

    else:
        return "Sorry I do not understand, please try again"


def chat():
    while True:
        user = input("You: ")
        if user.lower() == "quit":
            break

        print(chatWithBot(user))