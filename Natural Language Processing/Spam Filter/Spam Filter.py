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

class spam_filter(object):
    
    def __init__(self, h, s, splt):
        self.ham_train, self.ham_test = self.list_txt(h, splt, pn = 'ham')
        self.spam_train, self.spam_test = self.list_txt(s, splt, pn = 'spam')
        
        self.l_test = self.ham_test + self.spam_test
        random.shuffle(self.l_test)
        
        self.l_train = self.ham_train + self.spam_train
        random.shuffle(self.l_train)
        
        self.b_test, self.a_test = self.extract_parts(self.l_test)
        self.b_train, self.a_train = self.extract_parts(self.l_train)
        
        self.b_test = self.words(self.b_test)
        self.b_train = self.words(self.b_train)
        self.a_test = self.address(self.a_test)
        self.a_train = self.address(self.a_train)
        
        self.t_wfreq = self.word_freq((self.b_train + self.b_test))
    
    #Import emails and create train and test sets
    def list_txt(self, path, split, pn): 
        #split = percentage of reviews as training data
        #path = path of the files
        
        reviews = []
        
        for i, filename in enumerate(os.listdir(path)): 
            reviews.insert(i, (open(path + filename, 'rb').read(), pn))
        
        #Define the splitting point
        p = int(len(reviews) * split) 
        
        train = reviews[0:p]
        test = reviews[p:len(reviews)]
        
        return train, test
    
    #Extract email parts
    def extract_parts(self, texts):
        address = []
        body = []
    
        for text in texts:
            #Extract bodies
            body.append([str(re.search(b'(?m)^Subject: (.+)$', 
                                       text[0], re.DOTALL).group(1)), text[1]]) 
            #Extract addresses
            address.append([str(re.search(b'(?m)^From: (.*)', 
                                          text[0]).group(1)), text[1]]) 
            
        return body, address
    
    #Get Words of the texts
    def words(self, texts):
        #Define tokenizer
        tokenizer = nltk.tokenize.RegexpTokenizer('((?<=[^\w\s])\w(?=[^\w\s])|(\W))+', 
                                                  gaps = True) 
        #Define stopwords
        stopWords = set(nltk.corpus.stopwords.words('english') +\
                        list((' ', '\n', 'b'))) 
        #Define translator
        ttab = str.maketrans(string.punctuation, ' ' * len(string.punctuation)) 
        
        for text in texts:
            #Tokenize
            text[0] = tokenizer.tokenize(BeautifulSoup(text[0]).get_text()) 
            #Lower case & translate
            text[0] = [w.lower().translate(ttab) for w in text[0]]
            #Filter out numbers
            text[0] = [re.sub('\d+', ' ', w) for w in text[0]] 
            #Filter out stopwords
            text[0] = [w for w in text[0] if w not in stopWords]
            #Stemming
            text[0] = [nltk.stem.porter.PorterStemmer().stem(w) for w in text[0]] 
    
        return texts
    
    #Extract addresses of senders
    def address(self, addrs): 
        for address in addrs:
            try:
                address[0] = str(re.search(r'[\w\.-]+@[\w\.-]+', 
                                 address[0]).group(0))
            except:
                address[0] = 'Unknown'
                
        return addrs
    
    #Get the word frequencies
    def word_freq(self, texts): 
        all_words = []
        
        for text in texts:
            for words in text[0]:
                all_words.append(words)
        
        all_words = nltk.FreqDist(all_words)
    
        return all_words
    
    #Return the top words
    def word_feat(self, text, probdist, top = 1000): 
        word_features = []
        features = {}
    
        for word in probdist.most_common()[:top]:
            word_features.append(word[0])
            
        for word in word_features:
            features[word] = (word in text[0])
            
        return features
    
    #Create features
    def feat_create(self, train, top = 1000):
        feats = []
        
        for text in train:
            feats.append((self.word_feat(text[0], self.t_wfreq, top), text[1])) 

        return feats

    #Testing various number of features
    def tuner(self, feats):
        
        for i in feats:
            feats_train = self.feat_create(self.b_train, top = i)
            feats_test = self.feat_create(self.b_test, top = i)
            classifier = nltk.NaiveBayesClassifier.train(feats_train)
            print("Testing Accuracy %d (NBC): " % (i), 
                  (nltk.classify.accuracy(classifier, feats_test)) * 100)

    #Ensemble with SKlearn & NLTK
    def classify(self, t):
        feats_train = self.feat_create(self.b_train, top = t)
        feats_test = self.feat_create(self.b_test, top = t)
        preds = np.chararray((len(feats_test), 5))
        
        classifier = nltk.NaiveBayesClassifier.train(feats_train)
        print("Testing Accuracy (NBC): ", 
              (nltk.classify.accuracy(classifier, feats_test)) * 100)
        for feats in range(len(feats_test)):
            preds[feats, 0] = str(classifier.classify(feats_test[feats][0]))
        
        classifier = nltk.SklearnClassifier(nb.BernoulliNB())
        classifier.train(feats_train)
        print("Testing Accuracy (BNB): ", 
              (nltk.classify.accuracy(classifier, feats_test)) * 100)
        for feats in range(len(feats_test)):
            preds[feats, 1] = str(classifier.classify(feats_test[feats][0]))
        
        classifier = nltk.SklearnClassifier(nb.MultinomialNB())
        classifier.train(feats_train)
        print("Testing Accuracy (MNB): ", 
              (nltk.classify.accuracy(classifier, feats_test)) * 100)
        for feats in range(len(feats_test)):
            preds[feats, 2] = str(classifier.classify(feats_test[feats][0]))
        
        classifier = nltk.SklearnClassifier(lm.LogisticRegression(C = 0.75))
        classifier.train(feats_train)
        print("Testing Accuracy (LR): ", 
              (nltk.classify.accuracy(classifier, feats_test)) * 100)
        for feats in range(len(feats_test)):
            preds[feats, 3] = str(classifier.classify(feats_test[feats][0]))
        
        classifier = nltk.SklearnClassifier(svm.NuSVC())
        classifier.train(feats_train)
        print("Testing Accuracy (NuSVC): ", 
              (nltk.classify.accuracy(classifier, feats_test)) * 100)
        for feats in range(len(feats_test)):
            preds[feats, 4] = str(classifier.classify(feats_test[feats][0]))
        
        result = []    
        for i in range(len(preds)):
            if (Counter(preds[i, :]).most_common()[0][0] == b's' and \
                self.b_test[i][1] == 'spam') or \
                (Counter(preds[i, :]).most_common()[0][0] == b'h' and \
                 self.b_test[i][1] == 'ham'):
                result.append(1)
            else:
                result.append(0)
        
        print('Total Result: %.2f' % (sum(result) / len(result) * 100))


if __name__ == "__main__":
    spam_filter = spam_filter(h = "F:/Code/Spam Filter/Ham/",
                              s = "F:/Code/Spam Filter/Spam/",
                              splt = 0.95)

    feats_1 = [3000, 3100, 3200, 3400, 3500, 3600, 3700, 3800, 3900]
    spam_filter.tuner(feats_1)
    spam_filter.classify(t = 3100)