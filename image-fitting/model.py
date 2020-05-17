import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense



class myModel(Model):
    def __init__(self):
        super(myModel, self).__init__()
        self.d1 = Dense(500, activation='relu')
        self.d2 = Dense(400, activation='relu')
        self.d3 = Dense(300, activation='relu')
        self.d4 = Dense(200, activation='relu')
        self.d5 = Dense(200, activation='relu')
        self.d6 = Dense(100, activation='relu')
        self.d7 = Dense(100, activation='relu')
        self.d8 = Dense(1)

    def call(self, x):
        x = self.d1(x)
        x = self.d2(x)
        x = self.d3(x)
        x = self.d4(x)
        x = self.d5(x)
        x = self.d6(x)
        x = self.d7(x)
        y = self.d8(x)
        return y
