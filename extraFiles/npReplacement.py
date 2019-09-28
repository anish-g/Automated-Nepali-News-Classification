# -*- coding: utf-8 -*-
"""
Created on Fri May 10 19:54:20 2019

@author: Binish125
"""

import codecs
import math
import os
import re
import numpy as np
from copy import deepcopy
from sklearn.svm import SVC

from sklearn.metrics import accuracy_score

#files containing stop words
Stop_word_file='nepali'
stop_words=np.empty((0))
decoded_stop_words=codecs.open(Stop_word_file,encoding='utf-8')
for line in decoded_stop_words:
    stop_words=np.append(stop_words,line.strip("\n").strip("\ufeff"))
decoded_stop_words.close()
del Stop_word_file;

    

class Tokenizer:
    
    tokens=np.empty((0));
    
    def makeTokens(self,sentence=""):
        self.tokens=np.array(sentence.split(" "))
        
    def remove_stop_words(self):
        new_tokens=np.empty((0))
        for index,token in enumerate(self.tokens):
            token = re.sub('\\r\\ufeff|\\r\\n\\ufeff\\n|-|’\\n|।\\n|\\n|\,|\"|\'| \)|\(|\)| \{| \}| \[| \]|!|‘|’|“|”| \:-|\?|।|/|\—', '', token)
            token = re.sub(r'[0-9०-९]','',token)
            token = re.sub(r'[a-zA-Z]','',token)
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
    
    #idf of the entire corpus
    corpus_idf={}
    
    #Y values 
    Y=np.empty((0))
    
    #tf-idf of the entire corpus dictionary
    corpus_tfidf=np.empty((0))
    
    #corpus tf-idf values only
    global tfidf
    
    #contains the word set for the entire corpous
    
    wordSet={}
    
    #contains the word dictionary (token : count) for the entire article set
    wordDict={}
    
    #contains the word counts of article wise
    wordDictList=[]
    
    #contains tokens sets of all articles
    tokens_set=[]
    
    def fit_transform(self,path):
        print("Reading Training Data-sets : ")
        
        for files in os.walk(path):
            cate_directory=files[0]
            
            #get the category from the folder name
            category_text=cate_directory.split("\\")[-1];
            if(category_text=='train' or category_text=='test'):
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
                    
                    
                    #set Y values for each article
                    self.Y=np.append(self.Y,category_val)
                    #self.Y.append(category_val)
                    
                    #add the unique article words to the words set
                    self.wordSet=set(self.tokenizer.get_tokens()).union(self.wordSet)
                    
                except:
                    continue;
        
        self.tokens_set=np.array(self.tokens_set);
        #delete tokens in tokenizer class
        self.tokenizer.del_tokens();
        del data;
        
        print("Traning Data loaded")
        #create a {word:0} dictionary for all words in the word set            
        self.wordDict={ key : 0 for key in self.wordSet }
        print("WordSet Created")
        
        del self.wordSet;
        
        #makes copies of the overall word dict for each article
        print("Deep Copying Dictionary (Length : " + str(len(self.tokens_set))+" )")
        for i in range(len(self.tokens_set)):
            #self.wordDictList=np.append(self.wordDictList,deepcopy(self.wordDict))
            self.wordDictList.append(deepcopy(self.wordDict))
        self.wordDictList=np.array(self.wordDictList)
        print("Computing TF-IDF")
        #count number of words in article and add to the article dict
        self.word_count()
        #compute TF 
        self.corpusTF()
        #compute IDF
        self.corpusIDF()
        #Compute TFIDF
        self.corpusTFIDF()
        print("TFIDF - completed")
    
    
    
    #iterate through each article token-set and count each token, add it into the
    #dict count of the article
    
    #count words in each article and maintain the count dictionary
    def word_count(self):
        for index,token_set in enumerate(self.tokens_set):
            for token in token_set:
                self.wordDictList[index][token]+=1
        print("Word Dictionary List Length : " + str(len(self.wordDictList)))
    #computing TF (Number of word appear in article / total number words in document)  
    
    #compute TF for an given article 
    
    #wordDict -> count dictionary for an article
    #    token_set -> tokens of the given article
    def computeTF(self,wordDict,token_set):
        tfDict={}
        tokenCount=len(token_set)
        for word,count in wordDict.items():
            tfDict[word]=count/float(tokenCount)
        return tfDict
        
    
    #compute TF of entire corpus -> feeds each article to computeTF function
    def corpusTF(self):
        tf_set=np.empty((0))
        print("Number of articles Read: " + str(self.numberOfarticle()))
        for index in range(self.numberOfarticle()):
            tf_set=np.append(tf_set,self.computeTF(self.wordDictList[index],self.tokens_set[index]))
            #tf_set.append(self.computeTF(self.wordDictList[index],self.tokens_set[index]))
        self.corpus_tf=tf_set
        print("TF-Corpus set ( Length : " + str(len(self.corpus_tf)) + ") ")
    
    
    #computing IDF (log(Number of documents/Number of documents that contains word w))
    
       # articleList->list of dictionary of word counts all the articles
    
    def computeIDF(self,articleList):
        idfDict={}
        NumberOfArticles=len(self.tokens_set)
        idfDict=dict.fromkeys(articleList[0].keys(),0)
        for article in articleList:
            for word,val in  article.items():
                if val>0:
                    idfDict[word] +=1
        for word,val in idfDict.items():
            idfDict[word]= math.log10(NumberOfArticles/float(val))
        self.corpus_idf=idfDict
    
    
    #get IDF of the corpus
    def corpusIDF(self):
        self.computeIDF(self.wordDictList)
        print("IDF-Corpus set ( Length : " + str(len(self.corpus_idf)) + ")")
        
    #get tfidf of the entire corpus
    def corpusTFIDF(self):
        self.tfidf=np.empty((0,len(self.corpus_idf)))
        for index,tf in enumerate(self.corpus_tf):
            tfidf={}
            tfidf_only=np.empty((0,len(tf)))
            for word,val in tf.items():
                tfidf[word]=val*self.corpus_idf[word]
                tfidf_only=np.append(tfidf_only,tfidf[word])
                #tfidf_only.append(tfidf[word])
            self.tfidf=np.vstack([self.tfidf,np.array(tfidf_only)])
            self.corpus_tfidf=np.append(self.corpus_tfidf,tfidf)
            
            #self.tfidf.append(tfidf_only)
            #self.corpus_tfidf.append(tfidf)
            
    #show tf-idf of an specific article (Extra)
    def getTFIDF(self,article_number):
        return( self.corpus_tfidf[article_number-1]  )  
   
    #get the total number of articles
    def numberOfarticle(self):
        return len(self.tokens_set)
   
    #tf-idf of an specific article -> containes {word:value}
    #def computeTFIDF(self,tf,idf):
    #    tfidf={}
    #    for word,val in tf.items():
    #        tfidf[word]=val * idf[word]
    #    return tfidf;
    
    
    #get an list of TFIDF values only ->  does not contain {word:value}
    def getTFIDFonly(self,tf,idf):
        tfidf_only=np.empty((0,0))
        for word,val in tf.items():
            tfidf_only=np.append(tfidf_only,val*idf[word])
            #tfidf_only.append(val*idf[word])
        return tfidf_only
    
    #returns corpus TFIDF values Only   
    def get_TFIDF(self):
        return self.tfidf;
    
    #returns Y values-List
    def get_Y(self):
        return self.Y;
    
    #transform each article into TF-IDF values
    def transform_article(self,article):
        token_set=np.empty((0))
        temp_wordSet=np.empty((0))
        tf_set=np.empty((0))
        self.tokenizer.makeTokens(article)
        self.tokenizer.remove_stop_words()
        
        token_set=np.append(token_set,self.tokenizer.get_tokens())
        #token_set.append(self.tokenizer.get_tokens())
        
        temp_wordSet=np.append(temp_wordSet,deepcopy(self.wordDict))
        #temp_wordSet.append(deepcopy(self.wordDict))
        
        #print(temp_wordSet[0])
        #print("Token Set here : " + str(token_set))
        for token in token_set[0]:
           try:
               temp_wordSet[0][token]+=1
           except:
               continue;
        #print("Number of articles : " + str(self.numberOfarticle()))
        
        tf_set=np.append(tf_set,self.computeTF(temp_wordSet[0],token_set))
        #tf_set.append(self.computeTF(temp_wordSet[0],token_set))
        
        article_TFIDF=self.getTFIDFonly(tf_set[0],self.corpus_idf)
        #print("Article_TFIDF length: " + str(len(article_TFIDF)))
        return article_TFIDF
        
    def transform(self,path):
        x_test=np.empty((0,len(self.corpus_idf)))
        y_test=np.empty((0))
        print("Reading Testing data")
        for files in os.walk(path):
            cate_directory=files[0]
            
            #get the category from the folder name
            category_text=cate_directory.split("\\")[-1];
            if(category_text=='test'):
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
                    #convert the data into tokens
                    article_tfidf=self.transform_article(data)
                    
                    x_test=np.vstack([x_test,article_tfidf])
                    #x_test.append(article_tfidf)
                    
                    y_test=np.append(y_test,category_val)
                    #y_test.append(category_val)
                    
                except:
                    continue
        print("Testing data load - completed")
        return x_test,y_test
    
    def show_DictList(self,article_num):
        print(self.wordDictList[article_num-1])
    
    #the wordset gets deleted once dictionary is created - to save memory
    #def show_wordSet(self):
    #    print(self.wordSet)
        
vectorizer=tfidfVectorizer()
vectorizer.fit_transform(".\\16NepaliNews\\16719\\train")
#tfidf=vectorizer.getTFIDF(1) #get the tfidf of an specific article

print("Loading datasets")
vectorized_x=vectorizer.get_TFIDF();
yTrain=vectorizer.get_Y();

print("Training model")
svm = SVC(kernel='linear')
svm.fit(vectorized_x,yTrain)
print("Model Trained Successfully");
#art_TFIDF=vectorizer.transform_article(" १३ पुस, काठमाडौं । पुस १२ गतेसम्ममा ८८.४० प्रतिशत सेयर अभौतिकीकरण (डिम्याट) भइसकेको नियाम निकाय नेपाल धितोपत्र बोर्डले जनाएको छ । नेपाल स्टक एक्सचेञ्जमा सूचीकृत २ सय ३२ कम्पनीहरुमध्ये १ सय २६ कम्पनीको सो परिणाम बराबरको सेयर अभौतिक भएको हो । बोर्डले अन्तिम पटक भन्दै गत मंसिर २० गते माघ १ गतेदेखि अभौतिक सेयरको मात्रै कारोबार हुने घोषणा गरेको थियो । बोर्डले निर्देश्न दिएपछि डिम्याट भएको सेयरको संख्या बढेको हो । रेजी, कारोबार रोक्का, निलम्बन, मर्जरमा रहेकाबाहेक जम्मा ५१ कम्पनीको धितोपत्र अभौतिकीकरण गर्न बाँकी रहेकोमा हाल थप २८ कम्पनीहरु अभौतिकरण प्रक्रियामा आईसकेको बोर्डका प्रवक्ता निरज गिरीले बताए । गिरीका अनुसार अब २३ सूचीकृत कम्पनीहरुको धितोपत्रमात्र अभौतिकीकरण हुन बाँकी रहेकोछ । अभौतिकीकरण प्रक्रियानि रन्तररुपमा चलिरहने प्रक्रिया भएकोले तत्काल सेयर खरीदबिक्री गर्नु नपर्ने लगानीकर्ताहरुले आफ्नो सहजतामा धितोपत्रअ भौतिकीकरण गर्न बोर्डले आहृवान गरेको छ । धितोपत्र बोर्डको निर्देशन अनुसार पुस मसान्तसम्ममा सबै सूचीकृत कम्पनीहरुको सेयर डिम्याट (अभौतिक) गरिसक्नुपर्ने छ । सम्बन्धित समाचारमाघ १ गतेबाट ‘डिम्याट’ सेयरको मात्रै कारोबार हुने")
print("\n\n\nTesting with Test data")
vectorized_y, ytest = vectorizer.transform(".\\16NepaliNews\\16719\\test");


print("Predicting Test Data")

y_pred=svm.predict(vectorized_y)
print("Prediciton completed")



print("ACcuracy : " + str(accuracy_score(y_pred,ytest)*100))

#cat_prediction = (svm.predict(art_TFIDF))
#if(cat_prediction[0]==0):
#    print("Predicted : Auto")
#elif(cat_prediction[0]==1):
#    print("Predicted : Bank")
