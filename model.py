import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense



class myModel(Model):
    def __init__(self):
        super(myModel, self).__init__()
        l2 = tf.keras.regularizers.l2(0.01)
        self.d1 = Dense(200, activation='relu', kernel_regularizer=l2)
        self.d2 = Dense(200, activation='relu', kernel_regularizer=l2)
        self.d3 = Dense(200, activation='relu', kernel_regularizer=l2)
        self.d4 = Dense(100, activation='relu', kernel_regularizer=l2)
        self.d5 = Dense(1, kernel_regularizer=l2)

    def call(self, x):
        x = self.d1(x)
        x = self.d2(x)
        x = self.d3(x)
        x = self.d4(x)
        y = self.d5(x)
        return y
