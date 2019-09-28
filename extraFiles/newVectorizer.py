# -*- coding: utf-8 -*-
"""
Created on Fri May 10 19:54:20 2019

@author: Binish125
"""

import codecs
import os
import re
import sys
import numpy as np
from sklearn.svm import SVC
from collections import Counter
from sklearn.metrics import accuracy_score

#np.set_printoptions(threshold=sys.maxsize)
class Tokenizer:
    
    tokens=np.empty((0));
    
    def makeTokens(self,sentence=""):
        self.tokens=np.array(sentence.split(" "))
        
    def remove_stop_words(self):
        new_tokens=np.empty((0))
        for index,token in enumerate(self.tokens):
            token = re.sub('\\r\\ufeff|\\r\\n\\ufeff\\n|-|’\\n|।\\n|\\n|\,|\"|\'| \)|\(|\)| \{| \}| \[| \]|!|‘|’|“|”| \:-|\?|।|/|\—', '', token)
            token = re.sub(r'[0-9०-९]','',token)
            #token = re.sub(r'[a-zA-Z]','',token)
            #stemming - removing मा|को|ले in words
            if re.findall(r'^.*(हरु|मा|को|ले|लाई|हरू|बाट|समेत|बीच|का|सहित|गरी|सँग|देखि|छैन|भरी)$', token):
                token = re.findall(r'^(.*)(?:हरु|मा|को|ले|लाई|हरू|बाट|समेत|बीच|का|सहित|गरी|सँग|देखि|छैन|भरी)$', token)
                token=token[0]
            if(token == ''):
                continue;
            elif(token not in stop_words):
                new_tokens=np.append(new_tokens,token)
        self.tokens=new_tokens
        del new_tokens
        
    def get_tokens(self):
        return self.tokens
    
    def show_tokens(self):
        print(self.tokens)
        
    def del_tokens(self):
        del self.tokens


class tfidfVectorizer:

    category={
                'Auto':0,
                'Bank':1,
                'Blog':2,
                'Business Interview':3,
                'Economy':4,
                'Education':5,
                'Employment':6,
                'Entertainment':7,
                'Interview':8,
                'Literature':9,
                'National News':10,
                'Opinion':11,
                'Sports':12,
                'Technology':13,
                'Tourism':14,
                'World':15,
                }

    tokenizer=Tokenizer()
    
    #TF of the entire corpus
    corpus_tf=np.empty((0))
    
    
    #Y values 
    Y=np.empty((0),dtype=np.uint8)
    
    
    #corpus tf-idf values only
    global tfidf
    
    global idf
    #contains the word set for the entire corpous
    
    wordSet=np.array([])
    
    #contains tokens sets of all articles
    tokens_set=[]
    
    
    #document Frequency dictinary 'word':{docIndex}
    DF={}
    
    #number of Documents
    N=0
    
    def get_TFIDF(self):
        return self.tfidf;
    
    def get_Y(self):
        return self.Y;
    
    def fit_transform(self,path):
        print("Reading Training Data-sets : ")
        
        for files in os.walk(path):
            cate_directory=files[0]
            
            #get the category from the folder name
            category_text=cate_directory.split("\\")[-1];
            if(category_text=='train' or category_text=='test' or category_text=='testRun' ):
                continue;
            print(" - " + category_text)
            
            #number-category representation
            category_val=self.category[category_text]
            
            del category_text
            del cate_directory
            
            # iterate through all articles
            for index,file in enumerate(files[2]):
                try:
                    #decode each file
                    decoded_file=codecs.open(files[0]+"\\"+file,encoding='utf-8')
                    #read the decoded file
                    data=decoded_file.read()
                    #convert the data into tokens
                    
                    decoded_file.close()
                    
                    self.tokenizer.makeTokens(data)
                    #remove stop words from the tokens
                    self.tokenizer.remove_stop_words()
                    
                    #append the token-set of the article to the article-tokens list
                    self.tokens_set.append(self.tokenizer.get_tokens())
                    #self.tokens_set.append(self.tokenizer.get_tokens())
                    
                    self.wordSet=set(self.wordSet).union(self.tokenizer.get_tokens())
                
                    #set Y values for each article
                    self.Y=np.append(self.Y,category_val)
                    #self.Y.append(category_val)
                    
                    #add the unique article words to the words set
                    #self.wordSet=set(self.tokenizer.get_tokens()).union(self.wordSet)
                    self.N=self.N+1
                except:
                    continue;
        file=open("numpySet.txt",'w',encoding='utf-8')
        file.write(str(self.wordSet))
        file.close()
        print("WordSet created")
        
        self.tokens_set=np.array(self.tokens_set);
        #delete tokens in tokenizer class
        self.tokenizer.del_tokens();
        del data;
        
        
        for i in range(len(self.tokens_set)):
            tokens=self.tokens_set[i]
            for token in tokens:
                try:
                    self.DF[token].add(i)
                except:
                    self.DF[token]={i}
        
        
        for i in self.DF:
            self.DF[i]=len(self.DF[i])
        
        #new wordset method
        #self.wordSet=np.append(self.wordSet,[word for word in self.DF])
        
        #print(self.wordSet)
        
        #print(self.DF)
        
        print("Total number of words : " + str(len(self.wordSet)))
        
        print("Total number of Documents in Corpus : " + str(self.N))
        
        self.tfidf=np.empty((0,len(self.wordSet)))
        
        idf=np.array([])
        for token in (self.wordSet):
            df = self.DF[token]
            #computing IDF (log(Number of documents/Number of documents that contains word w))
            idf=np.append(idf,np.log10((self.N/df)+1))
        self.idf=idf        
        print("Computing TFIDF")        
        for i in range(self.N):
            #print("Article " +  str(i) + " -> Y : " + str(self.Y[i]))
            #for each article
            tokens=self.tokens_set[i]
            counter=Counter(tokens)
            words_count=len(tokens)
            article_tfidf=np.empty((0,len(self.wordSet)))
            for index,token in enumerate(self.wordSet):
                #computing TF (Number of word appear in article / total number words in document)  
                tf = counter[token]/words_count
                
                tf_idf_only = tf*idf[index]
                article_tfidf=np.append(article_tfidf,tf_idf_only)
            self.tfidf=np.vstack([self.tfidf,article_tfidf])
        
    def transform(self,path):
        test_token_set=[]
        y_test=np.empty((0),dtype=np.uint8)
        print("Reading Testing data")
        for files in os.walk(path):
            cate_directory=files[0]
            
            #get the category from the folder name
            category_text=cate_directory.split("\\")[-1];
            if(category_text=='test' or category_text=='testRun' ):
                continue;
            #number-category representation
            category_val=self.category[category_text]
            
            # iterate through all articles
            for index,file in enumerate(files[2]):
                try:
                    #decode each file
                    decoded_file=codecs.open(files[0]+"\\"+file,encoding='utf-8')
                    #read the decoded file
                    data=decoded_file.read()
                    
                    #close the file once data is read
                    decoded_file.close()
                    
                    #make tokens of the article
                    self.tokenizer.makeTokens(data)
                    
                    #remove stop words from the tokens
                    self.tokenizer.remove_stop_words()
                    
                    #append the token-set of the article to the article-tokens list
                    test_token_set.append(self.tokenizer.get_tokens())
                                                                    
                    y_test=np.append(y_test,category_val)
                    #y_test.append(category_val)
                    
                except:
                    continue
                
        x_test=np.empty((0,len(self.wordSet)))
        #calculate TF-IDF
        self.N=len(test_token_set)
        print("Computing TFIDF")
        for i in range(self.N):
            #print("Article " +  str(i) + " -> Y : " + str(y_test[i]))
            #for each article
            tokens=test_token_set[i]
            counter=Counter(tokens)
            words_count=len(tokens)
            article_tfidf=np.empty((0,len(self.wordSet)))
            for index,token in enumerate(self.wordSet):
                #computing TF (Number of word appear in article / total number words in document)  
                tf = counter[token]/float(words_count)
                tf_idf_only = tf*self.idf[index]
                article_tfidf=np.append(article_tfidf,tf_idf_only)
            x_test=np.vstack([x_test,article_tfidf])
            
        print("Testing data load - completed")
        del test_token_set
       
        return x_test,y_test
        

#files containing stop words
Stop_word_file='nepali'
stop_words=np.empty((0))
decoded_stop_words=codecs.open(Stop_word_file,encoding='utf-8')
for line in decoded_stop_words:
    stop_words=np.append(stop_words,line.strip("\n").strip("\ufeff"))
decoded_stop_words.close()
del Stop_word_file;
        
vectorizer=tfidfVectorizer()
vectorizer.fit_transform(".\\16NepaliNews\\16719\\train")




print("Loading Training datasets")
vectorized_x=vectorizer.get_TFIDF();
yTrain=vectorizer.get_Y();

#print(vectorized_x)
#print(yTrain)


print("Training model")
svm = SVC(kernel='linear')
svm.fit(vectorized_x,yTrain)
print("Model Trained Successfully");

print("\n\nTesting with Test data")
vectorized_test, ytest = vectorizer.transform(".\\16NepaliNews\\16719\\test");

print("Predicting Test Data")
y_pred=svm.predict(vectorized_test)

print("Prediciton completed")
print("ACcuracy : " + str(accuracy_score(y_pred,ytest)*100))