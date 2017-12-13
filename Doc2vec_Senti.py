import keras
import numpy as np
from keras.datasets import imdb
from gensim.models import doc2vec
from collections import namedtuple
from keras.preprocessing import sequence
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
from sklearn.metrics import accuracy_score

def fit_with_SVM(train_vectors,train_labels,test_vectors,test_labels):
    #fitting with svm
    clf=svm.SVC()
    clf.fit(train_vectors,train_labels)
    test_pred=clf.predict(test_vectors)
    return (accuracy_score(test_labels,test_pred))

def fit_with_MLP(train_vectors,train_labels,test_vectors,test_labels):
    seed=7
    np.random.seed(seed)
    ###MLP with 500 units in one hidden layer
    model=Sequential();#model.add(Embedding(25000,50,input_length=300))
    model.add(Dense(500,activation='relu',input_shape=train_vectors.shape[1:]))
    model.add(Dense(1,activation='sigmoid'))
    model.compile(loss='binary_crossentropy',optimizer='rmsprop',metrics=['accuracy'])
    print(model.summary())
    #change X_train,X_test appropriately
    model.fit(train_vectors, train_labels, validation_data=(test_vectors, test_labels), epochs=5, batch_size=128, verbose=2)
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
    model.fit(train_vectors,train_labels,batch_size=30,epochs=5,validation_data=(test_vectors,test_labels))
    scores = model.evaluate(test_vectors, test_labels, verbose=0);
    print scores;
    print("Accuracy: %.2f%%" % (scores[1]*100))
    return scores[1]*100


def classify(train_vectors,train_labels,test_vectors,test_labels):
    #fitting with SVM
    #accuracy=fit_with_SVM(train_vectors,train_labels,test_vectors,test_labels)
    #print ("Accuracy with SVM using average is " + str(accuracy))
    #fitting with MLP
    accuracy=fit_with_MLP(train_vectors,train_labels,test_vectors,test_labels)
    print ("Accuracy with MLP using average is " + str(accuracy))
    #fitting with CNN
    accuracy=fit_with_CNN(train_vectors,train_labels,test_vectors,test_labels)
    print ("Accuracy with CNN using average is " + str(accuracy))


# Load data
INDEX_FROM=3
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=1000,index_from=INDEX_FROM,skip_top=10)
max_review_length=300
X_train=sequence.pad_sequences(X_train,maxlen=max_review_length)
X_test=sequence.pad_sequences(X_test,maxlen=max_review_length)
#print (type(X_train[0]))
word_to_id = keras.datasets.imdb.get_word_index()
word_to_id = {k:(v+INDEX_FROM) for k,v in word_to_id.items()}
word_to_id["<PAD>"] = 0
word_to_id["<START>"] = 1
word_to_id["<UNK>"] = 2
id_to_word = {value:key for key,value in word_to_id.items()}
#print(' '.join(id_to_word[id] for id in X_train[0] ))
def count_train_vectors():
    train_data=[]
    test_data=[]
    N=X_train.shape[0]
    for i in range(0,N):
        #print(' '.join(str(id) for id in X_train[i]))
        train_data.append((' '.join(id_to_word[id] for id in X_train[i])))
    N=X_test.shape[0]
    for i in range(0,N):
        test_data.append((' '.join(id_to_word[id] for id in X_train[i])))
    return train_data,test_data
train_data,test_data=count_train_vectors()

doc1 = ["This is a sentence", "This is another sentence"]

# Transform data (you can add more data preprocessing steps)

docs_train = []
analyzedDocument = namedtuple('AnalyzedDocument', 'words tags')
for i,text in enumerate(train_data):
    words = text.lower().split()
    tags = [i]
    #print(type(tags))
    docs_train.append(analyzedDocument(words, tags))

docs_test = []
analyzedDocument = namedtuple('AnalyzedDocument', 'words tags')
for i,text in enumerate(test_data):
    words = text.lower().split()
    tags = [i]
    #print(type(tags))
    docs_test.append(analyzedDocument(words, tags))

# for i, text in enumerate(doc1):
#     words = text.lower().split()
#     tags = [i]
#     print (tags)
#     print(type(tags))
#     docs.append(analyzedDocument(words, tags))

# Train model (set min_count = 1, if you want the model to work with the provided example data set)

model_train = doc2vec.Doc2Vec(docs_train, size = 100, window = 300, min_count = 1, workers = 4)
model_test = doc2vec.Doc2Vec(docs_test, size = 100, window = 300, min_count = 1, workers = 4)

# Get the vectors

print(np.array(model_train.docvecs).shape)
print(np.array(y_train).shape)
print(np.array(model_test.docvecs).shape)
print(np.array(y_test).shape)
#model.docvecs[1]
classify(np.array(model_train.docvecs),np.array(y_train),np.array(model_test.docvecs),np.array(y_test))
