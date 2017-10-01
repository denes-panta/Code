#Libraries
import string
import os
import nltk
import random
import re
from sklearn import naive_bayes as nb
from sklearn import linear_model as lm
from sklearn import svm as svm
import numpy as np
from bs4 import BeautifulSoup
from collections import Counter

#Functions
def list_txt(path, split = 0.95, pn = False):
    reviews = []
    
    for i, filename in enumerate(os.listdir(path)):
        reviews.insert(i, (open(path + filename, 'rb').read(), pn))
    
    p = int(len(reviews) * split)
    
    train = reviews[0:p]
    test = reviews[p:len(reviews)]   
    return train, test

def extract_parts(texts):
    address = []
    body = []

    for text in texts:
        body.append([str(re.search(b'(?m)^Subject: (.+)$', text[0], re.DOTALL).group(1)), text[1]])
        address.append([str(re.search(b'(?m)^From: (.*)', text[0]).group(1)), text[1]])
    return body, address

def words(texts):
    tokenizer = nltk.tokenize.RegexpTokenizer('((?<=[^\w\s])\w(?=[^\w\s])|(\W))+', gaps = True)
    stopWords = set(nltk.corpus.stopwords.words('english') + list((' ', '\n', 'b')))
    ttab = str.maketrans(string.punctuation, ' ' * len(string.punctuation))
    
    for text in texts:
        text[0] = tokenizer.tokenize(BeautifulSoup(text[0]).get_text())
        text[0] = [word.lower().translate(ttab) for word in text[0]]
        text[0] = [re.sub('\d+', ' ', word) for word in text[0]]
        text[0] = [word for word in text[0] if word not in stopWords]
        text[0] = [nltk.stem.porter.PorterStemmer().stem(word) for word in text[0]]
    return texts

def address(addrs):
    for address in addrs:
        try:
            address[0] = str(re.search(r'[\w\.-]+@[\w\.-]+', address[0]).group(0))
        except:
            address[0] = 'Unknown'
    return addrs

def word_freq(texts):
    all_words = []
    for text in texts:
        for words in text[0]:
            all_words.append(words)
    
    all_words = nltk.FreqDist(all_words)
    return all_words

def word_feat(text, probdist, top = 500):
    word_features = []
    features = {}

    for word in probdist.most_common()[:top]:
        word_features.append(word[0])
        
    for word in word_features:
        features[word] = (word in text[0])
    return features

def feat_create(train, top = 1000):
    feats = []
    
    for text in train:
        feats.append((word_feat(text[0], t_wfreq, top), text[1])) 
    return feats

#Script
#Pre-processing
ham_train, ham_test = list_txt("F:/Code/Spam Filter/Ham/", pn = 'ham')
spam_train, spam_test = list_txt("F:/Code/Spam Filter/Spam/", pn = 'spam')

l_test = ham_test + spam_test
random.shuffle(l_test)

l_train = ham_train + spam_train
random.shuffle(l_train)

b_test, a_test = extract_parts(l_test)
b_train, a_train = extract_parts(l_train)

del ham_test, spam_test, ham_train, spam_train, l_train, l_test

b_test = words(b_test)
b_train = words(b_train)
a_test = address(a_test)
a_train = address(a_train)

t_wfreq = word_freq((b_train + b_test))

#Testing various number of features
feats_1 = [100, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000]
feats_2 = [3000, 3100, 3200, 3400, 3500, 3600, 3700, 3800, 3900]

for i in feats_2:
    feats_train = feat_create(b_train, top = i)
    feats_test = feat_create(b_test, top = i)
    classifier = nltk.NaiveBayesClassifier.train(feats_train)
    print("Testing Accuracy %d (NBC): " % (i), (nltk.classify.accuracy(classifier, feats_test)) * 100)
    
#Ensemble with SKlearn & NLTK
feats_train = feat_create(b_train, top = 3100)
feats_test = feat_create(b_test, top = 3100)
preds = np.chararray((len(feats_test), 3))

classifier = nltk.NaiveBayesClassifier.train(feats_train)
print("Testing Accuracy %d (NBC): ", (nltk.classify.accuracy(classifier, feats_test)) * 100)
for feats in range(len(feats_test)):
    preds[feats, 0] = str(classifier.classify(feats_test[feats][0]))

classifier = nltk.SklearnClassifier(nb.BernoulliNB())
classifier.train(feats_train)
print("Testing Accuracy (BNB): " , (nltk.classify.accuracy(classifier, feats_test)) * 100)
for feats in range(len(feats_test)):
    preds[feats, 1] = str(classifier.classify(feats_test[feats][0]))

classifier = nltk.SklearnClassifier(nb.MultinomialNB())
classifier.train(feats_train)
print("Testing Accuracy (MNB): " , (nltk.classify.accuracy(classifier, feats_test)) * 100)
for feats in range(len(feats_test)):
    preds[feats, 2] = str(classifier.classify(feats_test[feats][0]))

classifier = nltk.SklearnClassifier(lm.LogisticRegression(C = 0.75))
classifier.train(feats_train)
print("Testing Accuracy (LR): " , (nltk.classify.accuracy(classifier, feats_test)) * 100)
for feats in range(len(feats_test)):
    preds[feats, 3] = str(classifier.classify(feats_test[feats][0]))

classifier = nltk.SklearnClassifier(svm.NuSVC())
classifier.train(feats_train)
print("Testing Accuracy (NuSVC): " , (nltk.classify.accuracy(classifier, feats_test)) * 100)
for feats in range(len(feats_test)):
    preds[feats, 4] = str(classifier.classify(feats_test[feats][0]))

result = []    
for i in range(len(preds)):
    if (Counter(preds[i, :]).most_common()[0][0] == b's' and b_test[i][1] == 'spam') or (Counter(preds[i, :]).most_common()[0][0] == b'h' and b_test[i][1] == 'ham'):
        result.append(1)
    else:
        result.append(0)

print(sum(result) / len(result) * 100)