# -*- coding: utf-8 -*-

from sklearn.externals import joblib
import re
import numpy as np
import codecs
import os

class Tokenizer:
    
    tokens=np.empty((0));
    
    def makeTokens(self,sentence=""):
        self.tokens=np.array(sentence.split(" "))
        
    def lemmatize(self,token):
        if re.findall(r'^.*(हरु|सम्म|मा|को|ले|दै|लाई|हरू|बाट|समेत|बीच|का|सहित|गरी|सँग|देखि|छैन|भरी)$', token):
                token = re.findall(r'^(.*)(?:हरु|दै|सम्म|मा|को|ले|लाई|हरू|बाट|समेत|बीच|का|सहित|गरी|सँग|देखि|छैन|भरी)$', token)
                token=token[0]
        return token
        
    def remove_characters(self,token):
        token = re.sub('\\r\\ufeff|\\r\\n\\ufeff\\n|-|’\\n|।\\n|\\n|\,|\"|\'| \)|\(|\)| \{| \}| \[| \]|!|‘|’|“|”| \:-|\?|।|/|\—\:', '', token)
        token = re.sub(r'[0-9०-९]','',token)
        token = re.sub(r'[a-zA-Z]','',token)
        return token   
    
    def remove_stop_words(self):
        new_tokens=np.empty((0))
        for index,token in enumerate(self.tokens):
            token=self.remove_characters(token)
            token=self.lemmatize(token)
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


class Vectorizer:
    
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
    
    wordList=[]
    
    idf=np.array([])
    
    tokenizer=Tokenizer()

    
    def get_category(self,art_id):
        for key,value in self.category.items():
            if(value==art_id[0]):
                return key
    
    
    #Reading the trained word list from file
    def get_wordList(self):
        wordList_path=os.path.dirname(__file__) + "/wordList/wordList.txt"
        wordListFile=open(wordList_path,'r',encoding='utf-8')
        data=wordListFile.readlines();
        for index,line in enumerate(data):
            word=line.split(":")
            if(word[0]=='\n'):
                continue;
            self.wordList.append(word[0])
            word=word[1].split("\n")
            self.idf=np.append(self.idf,word[0])
        del data
        
    def transform_article(self,article):
        sentences=article.split("।");
        token_set=[]
        wordDict={}
        for index,sentence in enumerate(sentences):
            sentences[index] = re.sub('\\r\\ufeff|\\r\\n\\ufeff\\n|-|’\\n|।\\n|\\n|\,|\"|\'| \)|\(|\)| \{| \}| \[| \]|!|‘|’|“|”| \:-|\?|।|/|\—', '', sentence)
        
        #make tokens of the article
        self.tokenizer.makeTokens(article)
        
        #remove stop words from the tokens
        self.tokenizer.remove_stop_words()
        
        #append the token-set of the article to the article-tokens list
        token_set.append(self.tokenizer.get_tokens())
        
        for each_word in self.wordList:
            wordDict[each_word]=0.0
        
        for index,tokens in enumerate(token_set):
            for each_token in tokens:
                try:
                    wordDict[each_token]+=1
                except:
                    continue;
        
        x_test=np.zeros((1,len(self.wordList)))
    
        
        print("\nComputing TFIDF-test Data")
        tokens=token_set[0]
        words_count=len(tokens)
        for index,token in enumerate(tokens):
            try:
                #computing TF (Number of word appear in article / total number words in document)  
                #tf = counter[token]/float(words_count)
                tf=wordDict[token]/float(words_count)
                wordindex=self.wordList.index(token)
                tf_idf_only = tf*float(self.idf[wordindex])
                x_test[0][wordindex]=tf_idf_only
            except:
                continue
            
        print("Article Preprocessing - completed")
        del token_set
        
        
        '''
        
        Text Summarization happens here
        
        '''
        
        print('--------------------------------------------------------')
        token_set=[]    
        
        for sentence in sentences:
            self.tokenizer.makeTokens(sentence)
            self.tokenizer.remove_stop_words()
            token_set.append(self.tokenizer.get_tokens())
        
        for index,each_word in enumerate(self.wordList):
            wordDict[each_word]=np.zeros((len(token_set)))
            
        for index,tokens in enumerate(token_set):
            for each_token in tokens:
                try:
                    wordDict[each_token][index]+=1
                except:
                    continue;
        
        sentences_tfidf=np.zeros((len(token_set),len(self.wordList)))
        
        for i in range(len(token_set)):
            #print("Article " +  str(i) + " -> Y : " + str(self.Y[i]))
            #for each article
            tokens=token_set[i]
            #counter=Counter(tokens)
            words_count=len(tokens)
            for index,token in enumerate(tokens):
                try:
                    #computing TF (Number of word appear in sentence / total number words in article)  
                    #tf = counter[token]/words_count
                    tf=wordDict[token][i]/words_count
                    wordindex=self.wordList.index(token)
                    tf_idf_only = tf*float(self.idf[wordindex])
                    sentences_tfidf[i][wordindex]=tf_idf_only
                except: 
                    continue;
        
        sentences_sum=np.empty((0))
        for sentence_tfidf in sentences_tfidf:
            sum_tf=np.sum(sentence_tfidf)
            sentences_sum=np.append(sentences_sum,sum_tf)
        
        
        total_sum=np.sum(sentences_sum)
        #print("Article TF-IDF total :" + str(total_sum))
        avg=total_sum/len(token_set);
        #print("Average TF-IDF total :" + str(avg))
        #top 3 picks
        #top_picks=sentences_sum.argsort()[-3:][::-1]
        #for picks in top_picks:
        #    print(sentences[picks])
        #print("\n\n")
        
        print("Summary : \n")
        
        for index,sentence in enumerate(sentences_sum):
            if sentence>=avg:
                print(sentences[index])
        
        print('--------------------------------------------------------')
        return x_test


#files containing stop words
Stop_word_file='nepali'
stop_words=np.empty((0))
decoded_stop_words=codecs.open(Stop_word_file,encoding='utf-8')
for line in decoded_stop_words:
    stop_words=np.append(stop_words,line.strip("\n").strip("\ufeff"))
decoded_stop_words.close()
del Stop_word_file;

vectorizer=Vectorizer()


linearPickle=os.path.dirname(__file__) + "/Pickles/linearPickle.pkl"
rbfPickle=os.path.dirname(__file__) + "/Pickles/rbfPickle.pkl"
randomPickle=os.path.dirname(__file__) + "/Pickles/randomForestPickle.pkl"

svm_linear=joblib.load(open(linearPickle,'rb'))
svm_rbf=joblib.load(open(rbfPickle,'rb'))
random_forest=joblib.load(open(randomPickle,'rb'))
vectorizer.get_wordList()
article=input("Enter Article :\t")
while(article!='Exit'):
    x_vectorized=vectorizer.transform_article(article)
    y_pred=svm_linear.predict(x_vectorized)
    print("SVM (Linear-Kernel) : " + str(vectorizer.get_category(y_pred)))
    y_pred=svm_rbf.predict(x_vectorized)
    print("SVM (rbf-kernel) : " + str(vectorizer.get_category(y_pred)))
    y_pred=random_forest.predict(x_vectorized)
    print("Random Forest Classifier : " + str(vectorizer.get_category(y_pred)))
    article=input("Enter Article :\t")

#print("Predicted Category : \t" + str(vectorizer.get_category(y_pred)))
