from django.shortcuts import render
import random 
import json
import pickle
import numpy as np
from django.http import JsonResponse
from django.http import HttpResponse


import nltk
from nltk.stem import WordNetLemmatizer

from tensorflow.keras.models import load_model

lemmatizer = WordNetLemmatizer()
intents = json.loads(open('intents.json').read())

words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbotmodel.h5')

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words


def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list

def get_response(intents_list, intents__json):
    tag = intents_list[0]['intent']
    list_of_intents = intents__json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result

print("Running :)")
print("\n")
print("\n")
print("\n")
print("\n")
print("\n")
print("\n")
'''
message = input("")
ints = predict_class(message)
res = get_response(ints, intents)
print("Kaiden: " + res)
print("\n")
'''
def index(request):
    return render(request, 'home.html', {})

def submit(request):
    if request.method == "POST":
        question = request.POST.get('Name')
        print(question)
        ints = predict_class(question)
        res = get_response(ints, intents)
        return HttpResponse(res)

    return render(request, 'home.html', {'res':res})




















