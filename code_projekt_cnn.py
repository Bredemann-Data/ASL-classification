#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 11:15:28 2023

@author: basti
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import random
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler

import tensorflow

from scikeras.wrappers import KerasClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D

from sklearn.utils import class_weight


from tensorflow.keras.models import load_model

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
import seaborn as sns

#%% Read Files
class Files():
    def __init__(self, file):
        self.data = pd.read_csv(file)
        self.pixels = self.data.loc[:, 'pixel1':]
        self.X = np.array(self.pixels).reshape(-1,28,28)
        self.labels = self.data['label']
        self.y = np.array(self.labels).reshape(-1,1)
       
#Check for missing values
    def check_nan(self):
        df_check = self.data[self.data.isna().any(axis=1)]
        if df_check.index.size == 0:
            print('no missing values in file')
        else:
            print(df_check)

# Check distribution of labels
    def check_distr(self):
        print(self.labels.value_counts())     
        
# Check pictures randomly
    def picture_plot(self):
        n = random.randint(0,len(self.X))
        picture= self.X[n]
        plt.suptitle(str(self.y[n]))
        plt.imshow(picture, cmap='Greys')
        pass

# check for duplicates of pictures
    def check_dupl(self):
        checker = self.data.drop_duplicates()
        duplicates = len(self.data) - len(checker)
        print("The Set contains %s duplicate rows." % duplicates)

# scaling and One Hot encoding
    def scale(self):
        X = self.X
        darkest = X.max()
        X_scaled = X/darkest
        return X_scaled

    def one_hot_encode(self):
        y = self.y
        oh = OneHotEncoder()
        oh = oh.fit(y)
        y_oh = oh.transform(y).toarray()
        return (oh, y_oh)
        
#%% Daten einlesen
train = Files('sign_mnist_train.csv')

#%% Nach Fehlern in Daten suchen

# Verteilung der labels checken 
train.check_distr()

# nach fehlenden Werten schauen
train.check_nan()

# nach doppelten Bilder schauen
train.check_dupl()

#%% Bilder anschauen

train.picture_plot()

#%% Scaling, One Hot Encoding

# scaling of X
X_train = train.scale()

# One Hot Encoding
oh, y_train = train.one_hot_encode()

#%% Funktionen für die Trainingsphase

# Erstellen eines Modells das an den Scikeras Wrapper übergeben werden kann
# Two convolutinal layers with a pool and a droput layer separating them.
def classifier_conv(conv1, conv2, dense1):
    def c():
       model = Sequential()
       model.add(Conv2D(conv1, kernel_size=(3, 3), activation='relu', 
                        input_shape=(28,28,1)))
       model.add(MaxPooling2D(pool_size=(2, 2)))
       model.add(Dropout(0.25))
       model.add(Conv2D(conv2, kernel_size=(3, 3), activation='relu'))
       model.add(MaxPooling2D(pool_size=(2, 2)))
       model.add(Dropout(0.25))
       # Reshape 2-Dim input for fully conncted NN
       model.add(Flatten())
       model.add(Dense(dense1, activation='relu'))
       model.add(Dense(24, activation='softmax'))
       model.compile(loss=tensorflow.keras.losses.categorical_crossentropy,
                     optimizer=tensorflow.keras.optimizers.Adam(),
                     metrics=['accuracy'])
       return model
    return c

# erweiterte Fit_methode mit erstellen des Wrapers, Early-Stopping und 
# Klassengewichten, sowie speichern des Modells
def fitting(network, name):    
    stop = tensorflow.keras.callbacks.EarlyStopping(monitor='val_accuracy', 
                                                        patience=10,
                                                        min_delta=0.001,
                                                        verbose=1)
    estimator = KerasClassifier(network, epochs=50, 
                                batch_size=50, 
                                verbose=1, 
                                validation_split=0.2,
                                class_weight='balanced',
                                callbacks=[stop])
    
    #Klassengewichte erstellen
    y_integers = np.argmax(y_train, axis=1)
    class_weights = class_weight.compute_class_weight('balanced', 
                                                      classes = np.unique(y_integers), 
                                                      y=y_integers)
    d_class_weights = dict(enumerate(class_weights))
    
    estimator.fit(X_train, y_train, class_weight = d_class_weights)
    estimator.model_.save(name)
    return estimator
    
# Lernkurve plotten 
def plotting(model, name):
    sns.set_style("white")
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize= (18, 6))
    fig.suptitle("Training Performance " + name)
    ax1.set(title='loss per epoch',
            ylabel = 'loss',
            xlabel = 'epoch',
            xlim = (0,50))
    ax2.set(title='accuracy per epoch',
            ylabel = 'accuracy',
            xlabel = 'epoch',
            xlim = (0,50))
    ax1.plot(model.history_['loss'], label='training loss')
    ax1.plot(model.history_['val_loss'], label='validation loss')
    ax1.legend()
    ax2.plot(model.history_['accuracy'], label='training accuracy')
    ax2.plot(model.history_['val_accuracy'], label='validation accuracy')
    ax2.legend()
    plt.savefig(name+'.png')
    plt.show()
    plt.close()

#Trainieren, speichern und plotten
def main(conv1, conv2, dense1):
    model_name = str(conv1)+'-'+str(conv2)+'-'+str(dense1)
    clf = classifier_conv(conv1, conv2, dense1)
    print(clf().summary())
    history = fitting(clf, model_name)
    plotting(history, model_name)
    return history

#%% Ausführen von Main() mit verschiedenen Hyperparametern
net1 = main(20, 40, 160)
net2 = main(30,60,240)
net3 = main(40, 80, 320)
# net4 = main(50,100,400)

#%% Lernkurven zusammen Plotten

sns.set_style("white")
fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize= (18, 6))
fig.suptitle("Training Performance of all CNNs in comparison")
ax1.set(title='loss per epoch',
        ylabel = 'loss',
        xlabel = 'epoch',
        xlim = (0,30))
ax2.set(title='accuracy per epoch',
        ylabel = 'accuracy',
        xlabel = 'epoch',
        xlim = (0,30))
ax1.plot(net1.history_['val_loss'], label='20, 40, 160')
ax1.plot(net2.history_['val_loss'], label='30,60,240')
ax1.plot(net3.history_['val_loss'], label='40, 80, 320')
ax1.legend()
ax2.plot(net1.history_['val_accuracy'], label='20, 40, 160')
ax2.plot(net2.history_['val_accuracy'], label='30,60,240')
ax2.plot(net3.history_['val_accuracy'], label='40, 80, 320')
ax2.legend()
plt.savefig('combined-train.png')
plt.show()
plt.close()
#%%             

                ###########     Testen ##########


#%% Load test file
test = Files('sign_mnist_test.csv')

#%% Fehler in Daten

# Check label Distribution
test.check_distr()

# nach fehlenden Werten schauen
test.check_nan()

# nach doppelten Bilder schauen
test.check_dupl()

#%% Check pictures

test.picture_plot()

#%% Skalieren

X_test = test.X/255

y_test = oh.transform(test.y).toarray()

#%% Evaluations_klasse
class Evaluation():
    
    def __init__(self,name):
        self.evaluator = load_model(name)
        self.name = name
    
    def prediction(self, X):
        y_pred = self.evaluator.predict(X)
        y_pred = np.argmax(y_pred, axis=1)
        return y_pred
    
    def heat(self,hmap):
        sns.set()
        sns.set_theme(style="darkgrid")
        sns.axes_style("whitegrid")
        sns.set_palette("husl")
        sns.set(font_scale= 0.8)
        
        fig, ax = plt.subplots(figsize=(9,6))
        plt.suptitle("Heatmap for test results")
        
        sns.heatmap(hmap, 
                    cmap=sns.cubehelix_palette(as_cmap=True), 
                    cbar=True,
                    annot=False,
                    linewidth=.5,
                    ax = ax
                    )
        
        # Vermeidung von Überschneidungen
        plt.tight_layout()

        # Speichern der Plots
        plt.savefig(self.name + "_heatmap.png")
        plt.show()
        plt.cla()
        plt.clf()
        plt.close()
    
    def confusion(self,X):
        y_pred = self.prediction(X)
        y_true = np.argmax(y_test, axis=1)
        matrix = confusion_matrix(y_true, y_pred, normalize="true")
        self.heat(matrix)
        return matrix
        
        
    
#%% Modell laden
test_model = Evaluation("20-40-160")

#%% Heatmap and Confusion Matrix

performance = test_model.confusion(X_test)

print(performance)

#%% Modell evaluieren
score = test_model.evaluator.evaluate(X_test, y_test)
print(score)

#%% class for subset of pictures

class Subset_Files(Files):
    def __init__(self, file, n):
        self.data = pd.read_csv(file)
        self.data = self.data[self.data.label == n]
        self.pixels = self.data.loc[:, 'pixel1':]
        self.X = np.array(self.pixels).reshape(-1,28,28)
        self.labels = self.data['label']
        self.y = np.array(self.labels).reshape(-1,1)

# picture comparison function
def pic_comp(a,b, title, ax1_title, ax2_title, figname):
    while True:
        sns.set_style("white")        
        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize= (12, 6))
        plt.suptitle(title)
        p1= a[random.randint(0,len(a))]
        p2= b[random.randint(0,len(b))]
        ax1.set_title(ax1_title)
        ax1.imshow(p1, cmap='Greys')
        ax2.set_title(ax2_title)
        ax2.imshow(p2, cmap='Greys')
        plt.savefig(figname + ".png")
        plt.show()
        happy = input("happy?" )
        if happy == "y":
            break
        else:
            continue

#%% Check for most common mistake: 18 als 22

#%%
train_22 = Subset_Files('sign_mnist_train.csv', 23)
train_18 = Subset_Files('sign_mnist_train.csv', 19)

pic_comp(train_22.X, train_18.X, "Comparison between 22 and 18", 
         "22 in train set", "18 in train set", "22train_18train")

#%%
train_22 = Subset_Files('sign_mnist_train.csv', 23)
test_18 = Subset_Files('sign_mnist_test.csv', 19)

pic_comp(train_22.X, test_18.X, "Comparison between 22 and 18", 
         "22 in train set", "18 in test set", "22train_18test")

#%%
train_18 = Subset_Files('sign_mnist_train.csv', 23)
test_22 = Subset_Files('sign_mnist_test.csv', 19)

pic_comp(train_18.X, test_22.X, "Comparison between 22 and 18", 
         "22 in test set", "18 in train set", "train18_test22")

#%% check for similarity of pictures within train set
train_5 = Subset_Files('sign_mnist_train.csv', 5)

# Visueller Vergleich
pic_comp(train_5.X, train_5.X, "Training data derived from same picture",
         "train set", "train set", "train_identical")

#%% check for similarity of pictures within test set
test_5 = Subset_Files('sign_mnist_test.csv', 5)

# Visueller Vergleich
pic_comp(test_5.X, test_5.X, "Testdata derived from same picture",
         "test set", "test set", "test_identical")

#%% check for similarity of pictures betwee train set and test set
train_5 = Subset_Files('sign_mnist_train.csv', 5)
test_5 = Subset_Files('sign_mnist_test.csv', 5)

# Visueller Vergleich
pic_comp(train_5.X, test_5.X, "Test and Trainingsdata derived from same picture",
         "train set", "test set", "test_train_identical")

#%% Test whether a letter is correctly recognized when not drawn from test or train set.

# Modulimport
from PIL import Image

# Bildimport
def external_img(org_file_name, conv_name, title):
    # Bild auf 28x28 skalieren
    img = Image.open(org_file_name).convert("L")
    img = img.resize((28,28))
    sns.set_theme(style="white")
    plt.suptitle(title)
    plt.imshow(img, cmap='Greys')
    plt.savefig(conv_name)
    #Bild dem Modell zur Vorhersage geben
    img = np.asarray(img).reshape(1,28,28)
    img = img/255
    pred = test_model.prediction(img)
    print("Prediction for image: ", pred)

    
#%% 

for num, letter in enumerate(["a", "b", "c", "d", "e"]):
    subset_train = Subset_Files('sign_mnist_train.csv', num)
    subset_test = Subset_Files('sign_mnist_test.csv', num)
    pic_comp(subset_train.X, subset_test.X, "letter " 
             + letter.upper() + " in trainings and test set",
             "train set", "test set", "letter_" + letter + "mnist")
    external_img("letter_"+letter+'.png', "letter_"+letter+'_ext', "Letter " 
                 + letter.upper() + " (not in test or train set)")
