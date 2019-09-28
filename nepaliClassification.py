# -*- coding: utf-8 -*-
"""
Created on Fri May 10 19:54:20 2019

"""

import codecs
import os
import re
import numpy as np
from sklearn.svm import SVC
#from collections import Counter
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib
import statistics
from sklearn.ensemble import RandomForestClassifier
from copy import deepcopy
from sklearn.naive_bayes import MultinomialNB
import time

#np.set_printoptions(threshold=sys.maxsize)
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
        token = re.sub('\\r\\ufeff|\\r\\n\\ufeff\\n|-|’\\n|।\\n|\\n|\,|\"|\'| \)|\(|\)| \{| \}| \[| \]|!|‘|’|“|”| \:-|\?|।|/|\—', '', token)
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
    
    wordlist=[]
    
    #wordDictinoary
    wordDict={} 
     
    #contains tokens sets of all articles
    tokens_set=[]
    
    
    #document Frequency dictinary 'word':{docIndex}
    DF={}
    
    #number of Documents
    N=0
    
    def get_category(self,art_id):
        for key,value in self.category.items():
            if(value==art_id[0]):
                return key
    
    def get_TFIDF(self):
        return self.tfidf;
    
    def get_Y(self):
        return self.Y;
    
    def fit_transform(self,path):
        print('--------------------------------------------------------')
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
                    
                    self.wordSet=set(self.wordSet).union(self.tokenizer.get_tokens())
                
                    #set Y values for each article
                    self.Y=np.append(self.Y,category_val)
                    self.N=self.N+1
                except:
                    continue;
                    
        print("\nWordSet Created\n")   
         
        print('--------------------------------------------------------')
        
        '''
           Calculating Document Frequency
        
        '''
        
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
        
        '''
        
            Word Set Filtering
        
        '''
        print("Filtering WordSet")
        
        df_min=10
        df_max=0.5
        temp_wordSet=deepcopy(self.wordSet)
        print("Length of wordSet Before : "  + str(len(self.wordSet)))
        for word in temp_wordSet:
            if(df_min>0):
                if(self.DF[word]<df_min):
                    self.wordSet.remove(word)
            else:
                print("The df_min value should be greater than 0")
        
            word_doc_percent=(self.DF[word]/self.N)
            if(word_doc_percent>df_max):
                self.wordSet.remove(word)
        
        print("Length of wordSet After : "  + str(len(self.wordSet)))
        del temp_wordSet
        
            
        '''
        
        Managing word Dictionary // futher implementation
        
        '''
        
        print("Creating Word Dictionary")
        for index,each_word in enumerate(self.wordSet):
            self.wordDict[each_word]=np.zeros((self.N))
            
            
        print("\nManaging WordCount\n")
        
        for index,tokens in enumerate(self.tokens_set):
            for each_token in tokens:
                try:
                    self.wordDict[each_token][index]+=1
                except:
                    continue;
        
        print("WordCount Finished\n")        
        
        
        print('--------------------------------------------------------')
        
        self.tokens_set=np.array(self.tokens_set);
        #delete tokens in tokenizer class
        self.tokenizer.del_tokens();
        del data;
        
        
        print("Total number of words : " + str(len(self.wordSet)))
        
        print("Total number of Documents in Corpus : " + str(self.N))
        
        self.tfidf=np.empty((0,len(self.wordSet)))
        
        idf=np.array([])
        for token in (self.wordSet):
            df = self.DF[token]
            #computing IDF (log(Number of documents/Number of documents that contains word w))
            idf=np.append(idf,np.log10((self.N/df)+1))
        self.idf=idf        
        
        '''
            Computing TFIDF-Training Data
        '''
        
        start=time.time()
        
        self.tfidf=np.zeros((self.N,len(self.wordSet)))
        self.wordlist=list(self.wordSet)
        
        #create wordList file : if found delete existing one and create new
        
        wordList_path=os.path.dirname(__file__) + "/wordList/wordList.txt"
        
        if(os.path.exists(wordList_path)):
           os.remove(wordList_path)
        
        wordListFile=open(wordList_path,'a',encoding='utf-8')
        
        print("Creating WordList File")
        for index,word in enumerate(self.wordlist):
            wordListFile.write(word+":" + str(idf[index]) + "\n")
        
        wordListFile.close()
        print("WordList-File Created")
        
        for i in range(self.N):
            #print("Article " +  str(i) + " -> Y : " + str(self.Y[i]))
            #for each article
            tokens=self.tokens_set[i]
            #counter=Counter(tokens)
            words_count=len(tokens)
            for index,token in enumerate(tokens):
                try:
                    #computing TF (Number of word appear in article / total number words in document)  
                    #tf = counter[token]/words_count
                    tf=self.wordDict[token][i]/words_count
                    wordindex=self.wordlist.index(token)
                    tf_idf_only = tf*idf[wordindex]
                    self.tfidf[i][wordindex]=tf_idf_only
                except:
                    continue;
        end=time.time()
        #del self.wordDict
        
        print("Time : " +str(end - start))
        
        print('--------------------------------------------------------')
        
    def transform(self,path):
        test_token_set=[]
        y_test=np.empty((0),dtype=np.uint8)
        
        #Read Test Data
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
    
        '''
            WordDict test data
        '''
    
        test_wordDict={}
    
        #print("Creating Word Dictionary")
        for index,each_word in enumerate(self.wordSet):
            test_wordDict[each_word]=np.zeros((self.N))
            
            
        #print("\nManaging WordCount\n")
        
        for index,tokens in enumerate(test_token_set):
            for each_token in tokens:
                try:
                    test_wordDict[each_token][index]+=1
                except:
                    continue;
        
        '''
            Computing TFIDF-Test Data
        '''
        
        x_test=np.zeros((self.N,len(self.wordSet)))
                
        #print("\nComputing TFIDF-test Data")
        for i in range(self.N):
            #print("Article " +  str(i) + " -> Y : " + str(y_test[i]))
            #for each article
            tokens=test_token_set[i]
            #counter=Counter(tokens)
            words_count=len(tokens)
            for index,token in enumerate(tokens):
                try:
                    #computing TF (Number of word appear in article / total number words in document)  
                    #tf = counter[token]/float(words_count)
                    tf=test_wordDict[token][i]/float(words_count)
                    wordindex=self.wordlist.index(token)
                    tf_idf_only = tf*self.idf[wordindex]
                    x_test[i][wordindex]=tf_idf_only
                except:
                    continue
            
        print("Testing data load - completed")
        del test_token_set
        
       
        return x_test,y_test
        

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
        
        for index,each_word in enumerate(self.wordSet):
            wordDict[each_word]=0.0
        
        for index,tokens in enumerate(token_set):
            for each_token in tokens:
                try:
                    wordDict[each_token]+=1
                except:
                    continue;
        
        x_test=np.zeros((1,len(self.wordSet)))
    
        
        print("\nComputing TFIDF-test Data")
        tokens=token_set[0]
        words_count=len(tokens)
        for index,token in enumerate(tokens):
            try:
                #computing TF (Number of word appear in article / total number words in document)  
                #tf = counter[token]/float(words_count)
                tf=wordDict[token]/float(words_count)
                tf_idf_only = tf*self.idf[index]
                wordindex=self.wordlist.index(token)
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
        
        for index,each_word in enumerate(self.wordSet):
            wordDict[each_word]=np.zeros((len(token_set)))
            
        for index,tokens in enumerate(token_set):
            for each_token in tokens:
                try:
                    wordDict[each_token][index]+=1
                except:
                    continue;
        
        sentences_tfidf=np.zeros((len(token_set),len(self.wordSet)))
        
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
                    wordindex=self.wordlist.index(token)
                    tf_idf_only = tf*self.idf[wordindex]
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
        std_dev=statistics.stdev(sentences_sum)
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
        
vectorizer=tfidfVectorizer()
vectorizer.fit_transform(".\\16NepaliNews\\16719\\train")

print("Loading Training datasets")
vectorized_x=vectorizer.get_TFIDF();
yTrain=vectorizer.get_Y();

#print(vectorized_x)
#print(yTrain)

print("\n\nTransforming Testing Datasets")
vectorized_test, ytest = vectorizer.transform(".\\16NepaliNews\\16719\\test");


print("Training models")
svm = SVC(kernel='rbf',gamma=0.1,C=100)
svm_linear = SVC(kernel='linear',gamma=0.1,C=100)
random_forest=RandomForestClassifier(n_estimators=100,criterion='entropy')

print("Fitting and trasforming the models");
svm.fit(vectorized_x,yTrain)
print("\nSVM (rbf-kernel) - trained")
svm_linear.fit(vectorized_x,yTrain)
print("\nSVM (linear-kernel) - trained")
random_forest.fit(vectorized_x,yTrain)
print("\nRandom Forest - trained")
multinomial=MultinomialNB()
multinomial.fit(vectorized_x,yTrain)
print("Multinomail Naive_Bayes trained")

print('--------------------------------------------------------')
print("Creating Model Pickle")

linearPickle=os.path.dirname(__file__)+'/Pickles/linearPickle.pkl'

rbfPickle=os.path.dirname(__file__)+'/Pickles/rbfPickle.pkl'

randomForestPickle=os.path.dirname(__file__)+'/Pickles/randomForestPickle.pkl'

mulitnomialPickle=os.path.dirname(__file__)+'/Pickles/multinomialPickle.pkl'

joblib.dump(svm_linear,open(linearPickle,'wb'))
joblib.dump(svm,open(rbfPickle,'wb'))
joblib.dump(random_forest,open(randomForestPickle,'wb'))

print("Predicting the Test Data")
y_pred=svm.predict(vectorized_test)
rbf_kernel=accuracy_score(y_pred,ytest)*100
y_pred=svm_linear.predict(vectorized_test)
linear_kernel =accuracy_score(y_pred,ytest)*100
y_pred=random_forest.predict(vectorized_test)
random_accuracy =accuracy_score(y_pred,ytest)*100
y_pred=multinomial.predict(vectorized_test)
multi_accuracy=accuracy_score(y_pred,ytest)*100
print("Calculating Accuracy")

print("ACcuracy SVC Linear-kernel : " + str(linear_kernel))
print("ACcuracy SVC rbf-kernel : " + str(rbf_kernel))
print("ACcuracy random-Forest : " + str(random_accuracy))
print("ACcuracy mulitnomial-naiveBayes : " + str(multi_accuracy))


print("\n\n---------------------Predict Article-------------\n")

print("(Enter 'Exit' to quit)\n")


article=input("Enter Article:\t")
while(article!='Exit'):    
    x_vector=vectorizer.transform_article(article)
    y_svm=svm.predict(x_vector)
    y_linear=svm_linear.predict(x_vector)
    y_random=random_forest.predict(x_vector)
    print("SVM (rbf kernel ) : " + str(vectorizer.get_category(y_svm)))
    print("SVM (linear kernel ) : " + str(vectorizer.get_category(y_linear)))
    print("Random Forest : " + str(vectorizer.get_category(y_random)))
    print("\n")
    article=input("Enter Article")
