#coding:utf-8
import keras
from keras.models import Sequential, Model
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers import Input
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from PIL import Image
import glob,os
from keras.optimizers import Adam,SGD
from keras.utils import np_utils
import matplotlib.pyplot as plt
from keras.applications.vgg19 import VGG19

def plot_learning_graph(history, cnt,epoch):
    max_val_acc = max(history["val_acc"])
    plt.plot(np.arange(epoch)+1, history["acc"],label= 'acc')
    plt.plot(np.arange(epoch)+1, history["val_acc"],label= 'val_acc')
    plt.xlabel('Epoch')
    plt.ylim((0,1))
    plt.legend()
    plt.savefig('round '+str(cnt) +'_acc.png')
    
    plt.plot(np.arange(epoch)+1, history["loss"],label= 'loss')
    plt.plot(np.arange(epoch)+1, history["val_loss"],label= 'val_loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.savefig('round '+str(cnt) +'_loss.png')

def build_cnn_model():
        
    model = Sequential()

    model.add(Conv2D(32, (3, 3), padding='same',input_shape=X_train.shape[1:]))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(32))
    model.add(Activation('relu'))
    model.add(Dense(32))
    model.add(Activation('relu'))
    model.add(Dense(2))
    
    return model
def build_transfer_model():
    base_model=VGG19(weights='imagenet',include_top=False,
                 input_tensor=Input(X_train.shape[1:]))

    model = Sequential()
    model.add(Flatten(input_shape=base_model.output_shape[1:]))
    model.add(Dense(32))
    model.add(Activation('relu'))
    model.add(Dense(32))
    model.add(Activation('relu'))
    model.add(Dense(32))
    model.add(Activation('relu'))
    model.add(Dense(2))
    model = Model(input=base_model.input,output=model(base_model.output))    

    for layer in base_model.layers[:17]:
        layer.trainable=False

    return model


folder = ["0_2","1_2"]
image_size = 32


X = []
Y = []
for index, name in enumerate(folder):
    dir = "./" + name
    files = glob.glob(dir + "/*.png")
    for i, file in enumerate(files):
        image = Image.open(file)
        image = image.convert("RGB")
        image = image.resize((image_size, image_size))
        data = np.asarray(image)
        X.append(data)
        Y.append(index)

X = np.array(X)
Y = np.array(Y)
X = X.astype('float32')
X = X / 255.0
Y = np_utils.to_categorical(Y, 2)

X_train, X_test, y_train, y_test = train_test_split(X, Y,random_state=1,test_size=0.2)
model = build_transfer_model()
ave_acc  =[]
kf = KFold(n_splits=5, shuffle=False,random_state=None)
cnt = 0
epoch=30
'''
for train_index, eval_index in kf.split(X_train):
    x_tra, x_eval = X_train[train_index], X_train[eval_index]
    y_tra, y_eval = y_train[train_index], y_train[eval_index]
    cnt+=1
    es_cb = keras.callbacks.EarlyStopping(patience=0, verbose=1)

    model.compile(SGD(lr=0.00001),loss='categorical_crossentropy',metrics=['acc'])
    history = model.fit(x_tra,y_tra,epochs=epoch,verbose=0,batch_size=16,validation_data=(x_eval,y_eval)).history
    
    score = model.evaluate(X_test,y_test,verbose=1)
    print('Cross Validation %d round'%cnt)
    print('Test loss',score[0])
    print('Test accuracy',score[1])
    print('---------------------------------')
    ave_acc.append(score[1])
print(np.mean(ave_acc))
'''
model.compile(SGD(lr=0.001), loss='categorical_crossentropy',metrics=['acc'])

history = model.fit(X_train,y_train,epochs=epoch,verbose=1,batch_size=16,validation_data=(X_test,y_test)).history
'''    
    score = model.evaluate(X_test,y_test,verbose=1)
    print('Cross Validation %d round'%cnt)
    print('Test loss',score[0])
    print('Test accuracy',score[1])
    print('---------------------------------')
    ave_acc.append(score[1])
print(np.mean(ave_acc))
'''
