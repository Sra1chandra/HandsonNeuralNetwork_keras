import keras
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from keras.layers import Dense, Dropout, Activation
from sklearn import svm
from gensim import models
from keras.datasets import imdb
from matplotlib import pyplot
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers import GlobalMaxPooling1D
from sklearn.decomposition import PCA
from keras.preprocessing.text import Tokenizer

def fit_with_SVM(train_vectors,train_labels,test_vectors,test_labels):
    #fitting with svm
    clf=svm.SVC()
    clf.fit(train_vectors,train_labels)
    test_pred=clf.predict(test_vectors)
    return (accuracy_score(test_labels,test_pred))

def fit_with_MLP(train_vectors,train_labels,test_vectors,test_labels):
    seed=7
    np.random.seed(seed)
    print(train_vectors.shape)
    print(train_labels.shape)
    print(test_vectors.shape)
    print(test_labels.shape)
    ###MLP with 500 units in one hidden layer
    model=Sequential();#model.add(Embedding(25000,50,input_length=300))
    model.add(Dense(500,activation='relu',input_shape=train_vectors.shape[1:]))
    model.add(Dense(1,activation='sigmoid'))
    model.compile(loss='binary_crossentropy',optimizer='rmsprop',metrics=['accuracy'])
    print(model.summary())
    #change X_train,X_test appropriately
    model.fit(train_vectors, train_labels, epochs=5, batch_size=128, verbose=2)
    scores = model.evaluate(test_vectors, test_labels, verbose=0);print scores;return scores[1]*100
    # print("Accuracy: %.2f%%" % (scores[1]*100))

def fit_with_CNN(train_vectors,train_labels,test_vectors,test_labels):
    train_vectors=train_vectors[:,:,np.newaxis];
    #train_labels=train_vectors[:,:,np.newaxis];
    test_vectors=test_vectors[:,:,np.newaxis];
    #test_labels=test_vectors[:,:,np.newaxis];
    model=Sequential()
    model.add(Conv1D(250,3,padding='same',activation='relu',input_shape=train_vectors.shape[1:]))
    model.add(MaxPooling1D())
    model.add(Flatten())
    model.add(Dense(200,activation='relu'))
    model.add(Dense(1,activation='sigmoid'));
    model.compile(loss='binary_crossentropy',optimizer='adamax',metrics=['accuracy'])
    print(model.summary())
    model.fit(train_vectors,train_labels,batch_size=30,epochs=5)
    scores = model.evaluate(test_vectors, test_labels, verbose=0);
    print scores;
    print("Accuracy: %.2f%%" % (scores[1]*100))
    return scores[1]*100


def classify(train_vectors,train_labels,test_vectors,test_labels):
    #fitting with SVM
    accuracy=fit_with_SVM(train_vectors,train_labels,test_vectors,test_labels)
    print ("Accuracy with SVM using average is " + str(accuracy))
    #fitting with MLP
    #accuracy=fit_with_MLP(train_vectors,train_labels,test_vectors,test_labels)
    #print ("Accuracy with MLP using average is " + str(accuracy))
    #fitting with CNN
    #accuracy=fit_with_CNN(train_vectors,train_labels,test_vectors,test_labels)
    #print ("Accuracy with CNN using average is " + str(accuracy))

w = models.KeyedVectors.load_word2vec_format('./GoogleNews-vectors-negative300.bin', binary=True)
INDEX_FROM=3
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=1000,index_from=INDEX_FROM,skip_top=10)

max_review_length=300
X_train=sequence.pad_sequences(x_train,maxlen=max_review_length)
X_test=sequence.pad_sequences(x_test,maxlen=max_review_length)


word_to_id = keras.datasets.imdb.get_word_index()
word_to_id = {k:(v+INDEX_FROM) for k,v in word_to_id.items()}
word_to_id["<PAD>"] = 0
word_to_id["<START>"] = 1
word_to_id["<UNK>"] = 2

id_to_word = {value:key for key,value in word_to_id.items()}
print(' '.join(id_to_word[id] for id in X_train[0] ))

#writing mean
mean_train=np.zeros((25000,300))
mean_test=np.zeros((25000,300))
idf_train=np.zeros((25000,300))
idf_test=np.zeros((25000,300))
#count_train=np.zeros((25000,300))
#count_test=np.zeros((25000,300))
def mean_train_vectors():
	feature_count=0;
	for feature in X_train:
    		for id in feature:
        		if id<=2:
            			continue
        		try:
            			mean_train[feature_count]+=w[id_to_word[id]]
        		except:
				pass
            		#print(id_to_word[id])
            		#print("skipped "+str(id))
    		mean_train[feature_count]/=500
    		feature_count=feature_count+1

	feature_count=0;
	for feature in X_test:
    		for id in feature:
        		if id<=2:
            			continue
        		try:
            			mean_test[feature_count]+=w[id_to_word[id]]
        		except:
				pass
            		#print(id_to_word[id])
            		#print("skipped "+str(id))
    		mean_test[feature_count]/=300
    		feature_count=feature_count+1

	classify(mean_train,y_train,mean_test,y_test)
#mean_train_vectors()
def idf_train_vectors():
    ##### weighted IDF
    vectorizer=TfidfTransformer()
    vectorizer.fit_transform(X_train)
    feature_count=0
    for feature in X_train:
        vector=vectorizer.transform(feature)
        vector=vector.toarray()
        #print(vector)
        i=0;
        #Normalize=0;
        for id in feature:
            if id<=2:
                i=i+1;
                continue
            try:
    		#print(id);print(id_to_word[id]);print(vector[0][i])
            	idf_train[feature_count]+=vector[0][i]*w[id_to_word[id]]
    		#print(idf_train[feature_count])
    		#print(idf_train[feature_count])
    		#Normalize=Normalize+vector[i]
            except:
    		#print(id_to_word[id])
    		pass
            i=i+1;
        #idf_train[feature_count]/=Normalize;
        #print(idf_train[feature_count])
        feature_count=feature_count+1
    feature_count=0
    for feature in X_test:
        vector=vectorizer.transform(feature)
        vector=vector.toarray()
        i=0;
        for id in feature:
            if id<=2:
                i=i+1;
                continue
            try:
            	idf_test[feature_count]+=vector[0][i]*w[id_to_word[id]]
    		#Normalize=Normalize+vector[i]
            except:
    		pass
            i=i+1;
        #idf_test[feature_count]/=Normalize;
        #print(Normalize)
        feature_count=feature_count+1
    #print(mean_train[0])
    #print(vector)
    #print(idf_train[0])
    classify(idf_train,y_train,idf_test,y_test)
#idf_train_vectors()
def bag_of_words():
    print(len(x_train), 'train sequences')
    print(len(x_test), 'test sequences')

    num_classes = np.max(y_train) + 1
    print(num_classes, 'classes')
    max_words=1000
    print('Vectorizing sequence data...')
    tokenizer = Tokenizer(num_words=max_words)
    train = tokenizer.sequences_to_matrix(x_train, mode='count')
    test = tokenizer.sequences_to_matrix(x_test, mode='count')
    print('x_train shape:', train.shape)
    print('x_test shape:', test.shape)
    classify(train,y_train,test,y_test)

bag_of_words()

def count_train_vectors():
    train_data=[]
    test_data=[]
    N=X_train.shape[0]
    for i in range(0,N):
        #print(' '.join(str(id) for id in X_train[i]))
        train_data.append((' '.join(str(id) for id in X_train[i])))
    N=X_test.shape[0]
    for i in range(0,N):
        test_data.append((' '.join(str(id) for id in X_train[i])))

    ####bag of word frequencies as weights
    vectorizer=CountVectorizer()
    count_train=vectorizer.fit_transform(train_data)
    count_test=vectorizer.transform(test_data)
    #pca = PCA(n_components=300)
    #count_train=pca.fit_transform(np.array(count_train.toarray()).astype(np.float))
    #count_test=pca.fit_transform(np.array(count_test.toarray()).astype(np.float))
    count_train=np.array(count_train.toarray()).astype(np.float)/300
    count_test=np.array(count_test.toarray()).astype(np.float)/300
    #print(X_train[0])
    #sumi=0
    #for i in count_train[0]:
    #    print i
    #    sumi=sumi+i
    #print(sumi)
    classify(count_train,y_train,count_test,y_test)

#count_train_vectors()

def convolution_train_vectors():
    #####convolution of word vectors give review vector
    feature_count=0
    for feature in X_train:
        conv_train[feature_count]=w[id_to_word[feature[0]]];
        for id in feature:
            conv_train[feature_count]=np.convolve(conv_train[feature_count],w[id_to_word[id]],'same')
        conv_train[feature]/=500
        feature_count=feature_count+1;
    feature_count=0
    for feature in X_test:
        for id in feature:
            conv_train[feature_count]=np.convolve(conv_test[feature_count],w[id_to_word[id]],'same')
        conv_train[feature]/=500
        feature_count=feature_count+1;

    classify(conv_train,y_train,conv_test,y_test)
# convolution_train_vectors()
