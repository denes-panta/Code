#Libraries
import string
import os
import nltk
import random
import re

#Functions
def list_txt(path, split = 0.95, pn = False):
    #split = percentage of reviews as training data
    #path = path of the files
    #pn = positive = True, negative = False
    
    reviews = []
    tokenizer = nltk.tokenize.RegexpTokenizer(r'((?<=[^\w\s])\w(?=[^\w\s])|(\W))+', gaps = True) #Define the tokenizer
    
    for i, filename in enumerate(os.listdir(path)): #Import reviews and tokenize
        reviews.insert(i, [list(tokenizer.tokenize(open(path + filename).read())), pn])
    
    p = int(len(reviews) * split) #define the splitting point
    
    train = reviews[0:p] #Train data
    test = reviews[p:len(reviews)]  #Test data 
    
    return train, test #Return the matrices


def words(texts): #preprocessing
    stopWords = set(nltk.corpus.stopwords.words('english') + list((' ', '\n'))) #define the stopwords
    ttab = str.maketrans(string.punctuation, ' ' * len(string.punctuation)) #translate punctuations to ' '
    
    for text in texts:
        text[0] = [word.lower().translate(ttab) for word in text[0]] #lower case the words and translate punctuations
        text[0] = [re.sub(r'\d+', ' ', word) for word in text[0] if word] #filter out digits
        text[0] = [word for word in text[0] if word not in stopWords] #filter out the stopwords
        text[0] = [nltk.stem.porter.PorterStemmer().stem(word) for word in text[0]] #Stem the remaining words

    return texts

def word_freq(texts): #calculate the word frequencies
    all_words = []
    
    for text in texts:
        for words in text[0]:
            all_words.append(words)
    
    all_words = nltk.FreqDist(all_words)
    
    return all_words

def word_feat(text, probdist, top = 1000): #turn the words into features
    word_features = []
    features = {}

    for word in probdist.most_common()[:top]: #take the most common words
        word_features.append(word[0])
        
    for word in word_features:
        features[word] = (word in text[0])
            
    return features #return the features

def feat_create(train, top = 1000): #take the top 'top' features
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
feats_train = feat_create(l_train, top = 1000)
feats_test = feat_create(l_test, top = 1000)

#Classifier
classifier = nltk.NaiveBayesClassifier.train(feats_train) # define the classifier
print("Testing Accuracy (NBC): ", (nltk.classify.accuracy(classifier, feats_test)) * 100) #print the test accuracy
classifier.most_informative_features(25) #return the most inforamtive features
