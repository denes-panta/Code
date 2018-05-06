#Libraries
import string
import os
import nltk
import random
import re
import io

class sentiment(object):
    def __init__(self, p, sp, n):
        #Import reviews and create train & test data
        self.neg_train, self.neg_test = self.list_txt(p, split = sp, pn = 'neg')
        self.pos_train, self.pos_test = self.list_txt(n, split = sp, pn = 'pos')
    
        self.l_test = self.pos_test + self.neg_test
        random.shuffle(self.l_test)

        self.l_train = self.pos_train + self.neg_train
        random.shuffle(self.l_train)
        
        #Preprocessing    
        self.l_test = self.words(self.l_test)
        self.l_train = self.words(self.l_train)
        
        #Calculate the word frequencies
        self.t_wfreq = self.word_freq((self.l_train + self.l_test))

        #Turn the words into features & Take the top 'top' features
        self.feats_train = self.feat_create(self.l_train, top = 5000)
        self.feats_test = self.feat_create(self.l_test, top = 5000)

    def list_txt(self, path, split = 0.95, pn = False):
        #split = percentage of reviews as training data
        #path = path of the files
        #pn = positive = True, negative = False
        
        reviews = []
        
        #Define the tokenizer
        tokenizer = nltk.tokenize.RegexpTokenizer(
                r'((?<=[^\w\s])\w(?=[^\w\s])|(\W))+', gaps = True) 
        
        #Import reviews and tokenize
        for i, filename in enumerate(os.listdir(path)): 
            reviews.insert(
                    i, 
                    [list(tokenizer.tokenize(io.open(path + filename, "r", encoding = "utf-8").read())), pn]
                    )
        
        #Define the splitting point
        p = int(len(reviews) * split) 
        
        train = reviews[0:p]
        test = reviews[p:len(reviews)]
        
        return train, test

    #Preprocessing
    def words(self, texts): 
        
        #Define Stopwords
        stopWords = set(nltk.corpus.stopwords.words('english') + \
                        list((' ', '\n')))
        
        #Translate punctuations to ' '
        ttab = str.maketrans(string.punctuation, ' ' * len(string.punctuation)) 
        
        for text in texts:
            #lower case the words and translate punctuations
            text[0] = [w.lower().translate(ttab) for w in text[0]]
            #filter out digits
            text[0] = [re.sub(r'\d+', ' ', w) for w in text[0] if w] 
            #filter out the stopwords
            text[0] = [w for w in text[0] if w not in stopWords] 
            #Stem the remaining words
            text[0] = [nltk.stem.porter.PorterStemmer().stem(w) for w in text[0]] 
    
        return texts
    
    #Calculate the word frequencies
    def word_freq(self, texts): 
        all_words = []
        
        for text in texts:
            for words in text[0]:
                all_words.append(words)
        
        all_words = nltk.FreqDist(all_words)
        
        return all_words
    
    #Turn the words into features
    def word_feat(self, text, probdist, top = 1000): 
        word_features = []
        features = {}
        
        #Take the most common words
        for word in probdist.most_common()[:top]: 
            word_features.append(word[0])
            
        for word in word_features:
            features[word] = (word in text[0])
                
        return features
    
    #Take the top 'top' features
    def feat_create(self, train, top = 1000): 
        feats = []
        
        for text in train:
            feats.append((self.word_feat(text[0], self.t_wfreq, top), text[1])) 
        
        return feats

    #Classifier
    def classify(self):
        #Define the classifier
        classifier = nltk.NaiveBayesClassifier.train(self.feats_train) 
        #Print the test accuracy
        print("Testing Accuracy (NBC): ", 
              (nltk.classify.accuracy(classifier, self.feats_test)) * 100) 
        
        #return the most informative features
        return classifier.most_informative_features(25)

#Script
#Pre-processing
if __name__ == "__main__":
    analysis = sentiment(p = "F:\\Code\\Web Crawlers\\reviews\\negative\\",
                         n = "F:\\Code\\Web Crawlers\\reviews\\positive\\",
                         sp = 0.95)
    most_informative = analysis.classify()
    print(most_informative)