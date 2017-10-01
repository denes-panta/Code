#Libraries
import string
import os
import nltk
import random
import re

#Functions
def list_txt(path, split = 0.95, pn = False):
    reviews = []
    tokenizer = nltk.tokenize.RegexpTokenizer(r'((?<=[^\w\s])\w(?=[^\w\s])|(\W))+', gaps = True)
    
    for i, filename in enumerate(os.listdir(path)):
        reviews.insert(i, [list(tokenizer.tokenize(open(path + filename).read())), pn])
    
    p = int(len(reviews) * split)
    
    train = reviews[0:p]
    test = reviews[p:len(reviews)]   
    return train, test

def words(texts):
    stopWords = set(nltk.corpus.stopwords.words('english') + list((' ', '\n')))
    ttab = str.maketrans(string.punctuation, ' ' * len(string.punctuation))
    
    for text in texts:
        text[0] = [word.lower().translate(ttab) for word in text[0]]
        text[0] = [re.sub(r'\d+', ' ', word) for word in text[0] if word]
        text[0] = [word for word in text[0] if word not in stopWords]
        text[0] = [nltk.stem.porter.PorterStemmer().stem(word) for word in text[0]]
    return texts

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
neg_train, neg_test = list_txt("F:/Code/Sentiment/neg/", pn = 'neg')
pos_train, pos_test = list_txt("F:/Code/Sentiment/pos/", pn = 'pos')

l_test = pos_test + neg_test
random.shuffle(l_test)

l_train = pos_train + neg_train
random.shuffle(l_train)

del neg_test, pos_test, neg_train, pos_train

l_test = words(l_test)
l_train = words(l_train)

t_wfreq = word_freq((l_train + l_test))
feats_train = feat_create(l_train, top = 500)
feats_test = feats = feat_create(l_test, top = 500)

#Classifier
classifier = nltk.NaiveBayesClassifier.train(feats_train)
print("Testing Accuracy (NBC): ", (nltk.classify.accuracy(classifier, feats_test)) * 100)
classifier.most_informative_features(25)
