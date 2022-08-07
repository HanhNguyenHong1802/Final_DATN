from operator import mod

import keras
from keras.callbacks import (EarlyStopping, LearningRateScheduler,
                             ModelCheckpoint, ReduceLROnPlateau)
from keras.layers import (BatchNormalization, Dense, GlobalAveragePooling2D,
                          GlobalAveragePooling3D)
from keras.layers.convolutional import (AveragePooling2D, AveragePooling3D,
                                        Conv2D, Conv3D, MaxPooling3D)
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.layers.core import Activation, Dense, Dropout, Flatten, Reshape
from tensorflow.keras.layers import BatchNormalization
from keras.models import Sequential
from tensorflow.keras.optimizers import Adadelta
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import generic_utils, np_utils


class ModelLoader():

    def __init__(self,img_rows,img_cols,model_version,patch_size,nb_classes,weight_decay):

        self.img_rows = img_rows
        self.img_cols = img_cols
        self.model_version = model_version
        self.patch_size=patch_size
        self.nb_classes=nb_classes
        self.weight_decay=weight_decay
    
        # Loads the specified model
        if self.model_version == "v1":
            print('Loading 3dcnn + lstm Adadelta')            
            self.model = self.v1()
        elif self.model_version == "v2":
            print('Loading 3dcnn + lstm Adadelta')            
            self.model = self.v2()
        

        else:
            raise Exception('No model with name {} found!'.format(model_version))
        
    
    def v2(self):
        l2=keras.regularizers.l2
        model = Sequential()
        model.add(Conv3D(16,(3,3,3),
                                input_shape=(self.patch_size, self.img_cols, self.img_rows, 3),
                                activation='relu'))
        model.add(BatchNormalization())
        model.add(Conv3D(16,(3,3,3), strides=(1,1,1),padding='same', 
                            dilation_rate=(1,1,1), kernel_initializer='he_normal',
                            kernel_regularizer=l2(self.weight_decay),  
                             activation = 'relu'))
        model.add(MaxPooling3D(pool_size=(2,2,2)))
        model.add(MaxPooling3D(pool_size=(1, 2,2)))
        model.add(Conv3D(64,(3,3,3), strides=(1,1,1),padding='same', 
                            dilation_rate=(1,1,1), kernel_initializer='he_normal',
                            kernel_regularizer=l2(self.weight_decay), 
                             activation = 'relu'))
        model.add(MaxPooling3D(pool_size=(1, 2,2)))
        model.add(Conv3D(128,(3,3,3), strides=(1,1,1),padding='same', 
                            dilation_rate=(1,1,1), kernel_initializer='he_normal',
                            kernel_regularizer=l2(self.weight_decay),
                             activation = 'relu'))
        model.add(MaxPooling3D(pool_size=(1, 2, 2)))
        model.add(BatchNormalization())
        model.add(ConvLSTM2D(filters=64, kernel_size=(3,3),
                          strides=(1,1),padding='same',
                              kernel_initializer='he_normal', recurrent_initializer='he_normal',
                              kernel_regularizer=l2(self.weight_decay), recurrent_regularizer=l2(self.weight_decay),
                              return_sequences=True))
        model.add(GlobalAveragePooling3D())
        model.add(Dropout(0.5))
        model.add(Dense(self.nb_classes,kernel_initializer='normal',activation = 'softmax'))
        model.summary()
        optimizer =  Adadelta()
        model.compile(loss='categorical_crossentropy', 
              optimizer=optimizer,metrics=['acc'])
        model.summary()
        return model
    
    def v1(self):
        l2=keras.regularizers.l2
        model = Sequential()
        model.add(Conv3D(16,(3,3,3),
                                input_shape=(self.patch_size, self.img_cols, self.img_rows, 3),
                                activation='relu'))
        model.add(Conv3D(16,(3,3,3), strides=(1,1,1),padding='same', 
                            dilation_rate=(1,1,1), kernel_initializer='he_normal',
                            kernel_regularizer=l2(self.weight_decay),  
                             activation = 'relu'))
        model.add(MaxPooling3D(pool_size=(2,2,2)))
        model.add(MaxPooling3D(pool_size=(1, 2,2)))
        model.add(Conv3D(64,(3,3,3), strides=(1,1,1),padding='same', 
                            dilation_rate=(1,1,1), kernel_initializer='he_normal',
                            kernel_regularizer=l2(self.weight_decay), 
                             activation = 'relu'))
        model.add(MaxPooling3D(pool_size=(1, 2,2)))
        model.add(Conv3D(128,(3,3,3), strides=(1,1,1),padding='same', 
                            dilation_rate=(1,1,1), kernel_initializer='he_normal',
                            kernel_regularizer=l2(self.weight_decay),
                             activation = 'relu'))
        model.add(MaxPooling3D(pool_size=(1, 2, 2)))
        model.add(ConvLSTM2D(filters=64, kernel_size=(3,3),
                          strides=(1,1),padding='same',
                              kernel_initializer='he_normal', recurrent_initializer='he_normal',
                              kernel_regularizer=l2(self.weight_decay), recurrent_regularizer=l2(self.weight_decay),
                              return_sequences=True))
        model.add(GlobalAveragePooling3D())
        model.add(Dropout(0.5))
        model.add(Dense(self.nb_classes,kernel_initializer='normal',activation = 'softmax'))
        model.summary()
        optimizer =  Adadelta()
        model.compile(loss='categorical_crossentropy', 
              optimizer=optimizer,metrics=['acc'])
        model.summary()
        return model
   
