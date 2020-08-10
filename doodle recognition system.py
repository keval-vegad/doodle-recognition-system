# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 13:50:19 2020

@author: keval
"""


import numpy as np
import pandas as pd
# %matplotlib inline
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split, GridSearchCV


import tensorflow as tf

from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
# K.set_image_dim_ordering('th')
K.set_image_data_format('channels_first')

from keras.models import load_model
import cv2
import numpy as np

# load the data
dog = np.load('full_numpy_bitmap_dog.npy')
apple = np.load('full_numpy_bitmap_apple.npy')
baseball = np.load('full_numpy_bitmap_baseball.npy')
bicycle = np.load('full_numpy_bitmap_bicycle.npy')
book = np.load('full_numpy_bitmap_book.npy')
cake = np.load('full_numpy_bitmap_cake.npy')
cat = np.load('full_numpy_bitmap_cat.npy')
clock = np.load('full_numpy_bitmap_clock.npy')
flower = np.load('full_numpy_bitmap_flower.npy')
hand = np.load('full_numpy_bitmap_hand.npy')
mug = np.load('full_numpy_bitmap_mug.npy')
parrot = np.load('full_numpy_bitmap_parrot.npy')
pencil = np.load('full_numpy_bitmap_pencil.npy')
sandwich = np.load('full_numpy_bitmap_sandwich.npy')
saxophone = np.load('full_numpy_bitmap_saxophone.npy')
sheep = np.load('full_numpy_bitmap_sheep.npy')
tree = np.load('full_numpy_bitmap_tree.npy')
truck = np.load('full_numpy_bitmap_truck.npy')
num_7 = np.load('Multi-class classification/num_7.npy')
num_9 = np.load('num_9.npy')


# add a column with labels
dog = np.c_[dog, np.zeros(len(dog))]
apple = np.c_[apple, 1*np.ones(len(apple))]
baseball = np.c_[baseball, 2*np.ones(len(baseball))]
bicycle = np.c_[bicycle, 3*np.ones(len(bicycle))]
book = np.c_[book, 4*np.ones(len(book))]
cake = np.c_[cake, 5*np.ones(len(cake))]
cat = np.c_[cat, 6*np.ones(len(cat))]
clock = np.c_[clock, 7*np.ones(len(clock))]
flower = np.c_[flower, 8*np.ones(len(flower))]
hand = np.c_[hand, 9*np.ones(len(hand))]
mug = np.c_[mug, 10*np.ones(len(mug))]
parrot = np.c_[parrot, 11*np.ones(len(parrot))]
pencil = np.c_[pencil, 12*np.ones(len(pencil))]
sandwich = np.c_[sandwich, 13*np.ones(len(sandwich))]
saxophone = np.c_[saxophone, 14*np.ones(len(saxophone))]
sheep = np.c_[sheep, 15*np.ones(len(sheep))]
tree = np.c_[tree, 16*np.ones(len(tree))]
truck = np.c_[truck, 17*np.ones(len(truck))]
num_7 = np.c_[num_7, 18*np.ones(len(num_7))]
num_9 = np.c_[num_9, 19*np.ones(len(num_9))]


# store the label codes in a dictionary
label_dict = {0:'dog', 1:'apple', 2:'baseball', 3:'bicycle', 4:'book',
              5:'cake', 6:'cat', 7:'clock', 8:'flower', 9:'hand',
              10:'mug', 11:'parrot', 12:'pencil', 13:'sandwich', 14:'saxophone',
              15:'sheep', 16:'tree', 17:'truck', 18:'seven', 19:'nine'}


print(dog.shape)
print(apple.shape)
print(baseball.shape)
print(bicycle.shape)
print(book.shape)
print(cake.shape)
print(clock.shape)
print(flower.shape)
print(hand.shape)
print(mug.shape)
print(parrot.shape)
print(pencil.shape)
print(sandwich.shape)
print(saxophone.shape)
print(sheep.shape)
print(tree.shape)
print(truck.shape)
print(num_7.shape)
print(num_9.shape)



# Create the matrices for scikit-learn (5,000 images per class):
X = np.concatenate((dog[:10000,:-1], apple[:10000,:-1], baseball[:10000,:-1], bicycle[:10000,:-1], book[:10000,:-1], cake[:10000,:-1],cat[:10000,:-1], clock[:10000,:-1], flower[:10000,:-1],hand[:10000,:-1], mug[:10000,:-1], parrot[:10000,:-1], pencil[:10000,:-1], sandwich[:10000,:-1], saxophone[:10000,:-1], sheep[:10000,:-1], tree[:10000,:-1], truck[:10000,:-1], num_7[:5000,:-1], num_9[:5000,:-1]),axis=0).astype('float32') # all columns but the last

y = np.concatenate((dog[:10000,-1], apple[:10000,-1], baseball[:10000,-1], bicycle[:10000,-1], book[:10000,-1], cake[:10000,-1],cat[:10000,-1], clock[:10000,-1], flower[:10000,-1],hand[:10000,-1], mug[:10000,-1], parrot[:10000,-1], pencil[:10000,-1], sandwich[:10000,-1], saxophone[:10000,-1], sheep[:10000,-1], tree[:10000,-1], truck[:10000,-1], num_7[:5000,-1], num_9[:5000,-1]),axis=0).astype('float32') #  the last column

# y = np.concatenate((dog[:5000,-1], octopus[:5000,-1], bee[:5000,-1], hedgehog[:5000,-1], giraffe[:5000,-1]), axis=0).astype('float32') # the last column

X_train, X_test, y_train, y_test = train_test_split(X/255.,y,test_size=0.5,random_state=0)

#CNN
# one hot encode outputs
y_train_cnn = np_utils.to_categorical(y_train)
y_test_cnn = np_utils.to_categorical(y_test)
num_classes = y_test_cnn.shape[1]


# reshape to be [samples][pixels][width][height]
X_train_cnn = X_train.reshape(X_train.shape[0], 1, 28, 28).astype('float32')
X_test_cnn = X_test.reshape(X_test.shape[0], 1, 28, 28).astype('float32')


# define the CNN model
"""
https://www.geeksforgeeks.org/keras-conv2d-class/
kernel_size

This parameter determines the dimensions of the kernel. 
Common dimensions include 1×1, 3×3, 5×5, and 7×7 which can be passed as (1, 1), (3, 3), (5, 5), or (7, 7) tuples.
It is an integer or tuple/list of 2 integers, specifying the height and width of the 2D convolution window.
This parameter must be an odd integer
"""
# define the CNN model
def cnn_model():
    # create model
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=(1, 28, 28), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# build the model
model = cnn_model()
# Fit the model
model.fit(X_train_cnn, y_train_cnn, validation_data=(X_test_cnn, y_test_cnn), epochs=10, batch_size=200)
# Final evaluation of the model
model.summary()

#cnn:2
def cnn_model():
    # create model
    model2 = Sequential()
    model2.add(Conv2D(32, (3, 3), input_shape=(1, 28, 28), activation='relu'))
    model2.add(Conv2D(32, (3, 3), activation='relu'))
    model2.add(MaxPooling2D(pool_size=(2, 2)))
    model2.add(Dropout(0.2))
    
    model2.add(Conv2D(64, (3, 3), input_shape=(1, 28, 28), activation='relu'))
    model2.add(Conv2D(64, (3, 3), activation='relu'))
    model2.add(MaxPooling2D(pool_size=(2, 2)))
    model2.add(Dropout(0.2))
    
    model2.add(Flatten())
    model2.add(Dense(256, activation='relu'))
    model2.add(Dropout(0.4))
    
    model2.add(Dense(num_classes, activation='softmax'))
    # Compile model
    model2.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model2


# build the model
model2 = cnn_model()
# Fit the model
model2.fit(X_train_cnn, y_train_cnn, validation_data=(X_test_cnn, y_test_cnn), epochs=15, batch_size=300)
# Final evaluation of the model
model2.summary()
#cnn2


#time
import time

start = time.time()
print("Total time: ", time.time() - start, "seconds")
#/time
scores = model.evaluate(X_test_cnn, y_test_cnn, verbose=0)
print('Final CNN accuracy: ', scores[1])# 0.9089


img = cv2.imread('dog1.png', 0)
img = cv2.resize(img, (28,28))
img_not = cv2.bitwise_not(img)

# Create kernel
kernel = np.array([[0, -1, 0], 
                   [-1, 5,-1], 
                   [0, -1, 0]])

# Sharpen image
image_sharp = cv2.filter2D(img_not, -1, kernel)

# some_image = img.reshape(28, 28)
# type(some_image)
plt.imshow(image_sharp, cmap="binary")#cmap just indicate color map like black and white or pink or gray or other
plt.axis("off")
plt.show()
# img = cv2.resize(img,(28,28))
# img = np.reshape(img,[1,1,28,28])

pred5_2 = model.predict((image_sharp/255).reshape(1, 1, 28, 28).astype('float32'))[0]
value_dict_1 = list(pred5_2).index(max(pred5_2))
print("the given image could be---->> ",label_dict.get(value_dict_1))


model1 = load_model('CNN-Model-with-digits.h5')
model1.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
pred5_1 = model1.predict((image_sharp/255).reshape(1, 1, 28, 28).astype('float32'))[0]
value_dict_2 = list(pred5_1).index(max(pred5_1))
print("the given image could be---->> ",label_dict.get(value_dict_2))


# Predicting the Test set results
y_pred_cnn = model.predict_classes(X_test_cnn, verbose=0)

from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred_cnn))
#               precision    recall  f1-score   support

#          0.0       0.77      0.69      0.73      5005
#          1.0       0.96      0.96      0.96      4916
#          2.0       0.83      0.91      0.87      4890
#          3.0       0.95      0.97      0.96      5022
#          4.0       0.92      0.91      0.92      4996
#          5.0       0.96      0.90      0.93      4975
#          6.0       0.86      0.80      0.83      4964
#          7.0       0.96      0.94      0.95      4943
#          8.0       0.90      0.93      0.92      4986
#          9.0       0.95      0.91      0.93      5026
#         10.0       0.94      0.92      0.93      5118
#         11.0       0.84      0.82      0.83      5048
#         12.0       0.92      0.95      0.93      5081
#         13.0       0.85      0.91      0.88      5006
#         14.0       0.86      0.92      0.89      5014
#         15.0       0.88      0.89      0.89      5028
#         16.0       0.93      0.91      0.92      4916
#         17.0       0.93      0.94      0.93      4993
#         18.0       0.98      0.99      0.99      2533
#         19.0       0.98      0.99      0.99      2540

#     accuracy                           0.90     95000
#    macro avg       0.91      0.91      0.91     95000
# weighted avg       0.90      0.90      0.90     95000

#Use different scikit-learn algorithms to make predictions:
    
from sklearn.metrics import roc_curve, auc
y_pred_prob = model.predict_proba(X_test_cnn, verbose=0)
import scikitplot as skplt
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(len(label_dict.keys())):
    fpr[i], tpr[i], _ = roc_curve(y_test_cnn[:, i], y_pred_prob[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_test_cnn.ravel(), y_pred_prob.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

plt.figure()
lw = 2
plt.plot(fpr[2], tpr[2], color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[2])
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC for Doodle Drawing Recognition')
plt.legend(loc="lower right")
plt.show()
#Random Forest

clf_rf = RandomForestClassifier(n_jobs=-1, random_state=0)
clf_rf.fit(X_train, y_train)
print(clf_rf)
y_pred_rf = clf_rf.predict(X_test)
acc_rf = accuracy_score(y_test, y_pred_rf)
print ('Random forest accuracy: ',acc_rf)

# parameters = {'n_estimators': [10,20,40,60,80,100,120,140,160,180,200,220,240,260]}

parameters = {'n_estimators': [10,20,40,60,80,100,120,140]}
clf_rf = RandomForestClassifier(n_jobs=-1, random_state=0)
rf = GridSearchCV(clf_rf, parameters, n_jobs=-1)
rf.fit(X_train, y_train)

results = pd.DataFrame(rf.cv_results_)

results.sort_values('mean_test_score', ascending = False)

results.plot('param_n_estimators','mean_test_score')

clf_rf = RandomForestClassifier(n_estimators=120, n_jobs=-1, random_state=0) # n_estimators: number of trees in forest
clf_rf.fit(X_train, y_train)
y_pred_rf = clf_rf.predict(X_test)
acc_rf = accuracy_score(y_test, y_pred_rf)
print ('random forest accuracy: ',acc_rf)#random forest accuracy:  77.88%

from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred_rf))
#               precision    recall  f1-score   support

#           0.0       0.48      0.50      0.49      5005
#           1.0       0.89      0.91      0.90      4916
#           2.0       0.76      0.77      0.76      4890
#           3.0       0.85      0.90      0.88      5022
#           4.0       0.78      0.79      0.78      4996
#           5.0       0.73      0.80      0.76      4975
#           6.0       0.68      0.62      0.65      4964
#           7.0       0.88      0.86      0.87      4943
#           8.0       0.76      0.76      0.76      4986
#           9.0       0.78      0.70      0.74      5026
#         10.0       0.90      0.80      0.84      5118
#         11.0       0.68      0.67      0.68      5048
#         12.0       0.91      0.90      0.90      5081
#         13.0       0.68      0.66      0.67      5006
#         14.0       0.83      0.82      0.82      5014
#         15.0       0.68      0.74      0.71      5028
#         16.0       0.82      0.82      0.82      4916
#         17.0       0.75      0.79      0.77      4993
#         18.0       0.98      0.98      0.98      2533
#         19.0       0.97      0.99      0.98      2540

#     accuracy                           0.78     95000
#     macro avg       0.79      0.79      0.79     95000
# weighted avg       0.78      0.78      0.78     95000

from sklearn.metrics import confusion_matrix 
confusion_matrix(y_test, y_pred_rf)

#plot ROC cure
from sklearn.metrics import roc_curve, auc
y_pred_prob = clf_rf.predict_proba(X_test)
import scikitplot as skplt
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(len(label_dict.keys())):
    fpr[i], tpr[i], _ = roc_curve(y_test_cnn[:, i], y_pred_prob[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_test_cnn.ravel(), y_pred_prob.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

plt.figure()
lw = 2
plt.plot(fpr[2], tpr[2], color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[2])
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC for Random Forest')
plt.legend(loc="lower right")
plt.show()


#/prob ROC


#KNN
clf_knn = KNeighborsClassifier(n_jobs=-1)
clf_knn.fit(X_train, y_train)
print(clf_knn)
y_pred_knn = clf_knn.predict(X_test)
acc_knn = accuracy_score(y_test, y_pred_knn)
print ('KNN accuracy: ',acc_knn) #KNN accuracy:  0.78.19

parameters = {'n_neighbors': [1,3,5,7,9,11]}

clf_knn = KNeighborsClassifier(n_jobs=-1)
knn = GridSearchCV(clf_knn, parameters, n_jobs=-1)
knn.fit(X_train, y_train)

results_knn = pd.DataFrame(knn.cv_results_)

results_knn.sort_values('mean_test_score', ascending = False)

results_knn.plot('param_n_neighbors','mean_test_score');

# clf_knn = KNeighborsClassifier(n_jobs=-1)
# clf_knn.fit(X_train, y_train)
# y_pred_knn = clf_knn.predict(X_test)
# acc_knn = accuracy_score(y_test, y_pred_knn)
# print ('KNN accuracy: ',acc_knn)


from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred_knn))
#          0.0       0.58      0.50      0.53      5005
#          1.0       0.67      0.95      0.79      4916
#          2.0       0.79      0.52      0.63      4890
#          3.0       0.89      0.91      0.90      5022
#          4.0       0.85      0.82      0.84      4996
#          5.0       0.83      0.75      0.79      4975
#          6.0       0.81      0.54      0.65      4964
#          7.0       0.66      0.93      0.77      4943
#          8.0       0.82      0.77      0.80      4986
#          9.0       0.88      0.75      0.81      5026
#         10.0       0.86      0.86      0.86      5118
#         11.0       0.81      0.57      0.67      5048
#         12.0       0.71      0.95      0.81      5081
#         13.0       0.71      0.77      0.74      5006
#         14.0       0.85      0.85      0.85      5014
#         15.0       0.73      0.75      0.74      5028
#         16.0       0.84      0.84      0.84      4916
#         17.0       0.81      0.82      0.81      4993
#         18.0       0.90      0.98      0.94      2533
#         19.0       0.88      0.99      0.93      2540

#     accuracy                           0.78     95000
#    macro avg       0.79      0.79      0.78     95000
# weighted avg       0.79      0.78      0.78     95000


from sklearn.metrics import confusion_matrix 
confusion_matrix(y_test, y_pred_knn)


#load KNN
import pickle 



# Its important to use binary mode 
knnPickle = open('knnpickle_file', 'wb') 

# source, destination 
pickle.dump(clf_knn, knnPickle)                      


# load the model from disk
loaded_model = pickle.load(open('knnpickle_file', 'rb'))
result_knn = loaded_model.predict(X_test)


#ROC
from sklearn.metrics import roc_curve, auc
y_pred_prob = clf_knn.predict_proba(X_test)
import scikitplot as skplt
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(len(label_dict.keys())):
    fpr[i], tpr[i], _ = roc_curve(y_test_cnn[:, i], y_pred_prob[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_test_cnn.ravel(), y_pred_prob.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

plt.figure()
lw = 2
plt.plot(fpr[2], tpr[2], color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[2])
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC for KNN')
plt.legend(loc="lower right")
plt.show()
#/ROC


#plot sample images

def plot_samples(input_array, rows=4, cols=5, title=''):
    '''
    Function to plot 28x28 pixel drawings that are stored in a numpy array.
    Specify how many rows and cols of pictures to display (default 4x5).  
    If the array contains less images than subplots selected, surplus subplots remain empty.
    '''
    
    fig, ax = plt.subplots(figsize=(cols,rows))
    ax.axis('off')
    plt.title(title)

    for i in list(range(0, min(len(input_array),(rows*cols)) )):      
        a = fig.add_subplot(rows,cols,i+1)
        imgplot = plt.imshow(input_array[i,:784].reshape((28,28)), cmap='gray_r', interpolation='nearest')
        plt.xticks([])
        plt.yticks([])

sheep_not_cats = X_test[y_test > y_pred_cnn] # true: 1 (sheep), predicted: 0 (cat)
cats_not_dog = X_test[np.logical_and((y_test == 6),(y_pred_cnn == 0))] # true: 0 (cat), predicted: 1 (sheep)
plot_samples(cat, title='Sample cat drawings\n')
plot_samples(cats_not_dog, title = 'Cat drawings that were classified as dog\n')
#/plot