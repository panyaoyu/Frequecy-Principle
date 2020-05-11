import numpy as np
import tensorflow as tf
from model import myModel
from matplotlib import pyplot as plt
import os
import warnings
import random

warnings.filterwarnings("ignore")

rand = random.randint(0,100000)
print(rand)

model_path = './model/%d/' % rand
fig_path = './picture/%d/' % rand

if not os.path.exists(model_path):
    os.makedirs(model_path)

if not os.path.exists(fig_path):
    os.makedirs(fig_path)

freq_len = 20
train_size = 61
test_size = 201
x_left, x_right = -10, 10
epochs = 100000
lr = 2e-5

MSE = tf.keras.losses.MeanSquaredError()
optim = tf.keras.optimizers.Adam(learning_rate=lr)
# losses = tf.keras.met


def myFunc1(x):
    return np.sin(x) + np.sin(4*x)

def myFunc2(x):
    return 1 / (1+np.exp(-x))

def myFunc3(x):
    return x + np.power(x,2) + 3 * np.power(x,3)

def myFunc4(x):
    return np.sin(10*x) + np.sin(15*x)

def myFunc5(x):
    return np.sin(x) + np.sin(2*x) + 10 * np.sin(10*x)

def myFunc6(x):
    return np.sin(x)

def myFunc7(x):
    return np.sin(15*x)

def myFFT(x,freq_len=20):
    x = np.squeeze(x)
    x = np.fft.fft(x)
    fft_coef = x[range(freq_len)]
    return np.absolute(fft_coef)

x_train = np.reshape(np.linspace(-10, 10, train_size),[train_size,1])
y_train = myFunc7(x_train)
x_test = np.reshape(np.linspace(-10, 10, test_size),[test_size,1])
y_test = myFunc7(x_test)
print(x_train.shape)
print(y_test.shape)
fft_train = myFFT(y_train, freq_len)
fft_test = myFFT(y_test, freq_len)
# print(fft/61)
# plt.semilogy((fft+1e-5)/train_size,'ro-')
# plt.show()

# x_train = tf.convert_to_tensor(x, dtype=tf.float32) 
# y_train = tf.convert_to_tensor(y, dtype=tf.float32)

model = myModel()

for epoch in range(epochs):
    tmp = model.trainable_variables
    with tf.GradientTape() as tape:
        y_pred = model(x_train)
        loss = MSE(y_train, y_pred) + model.losses
    grads = tape.gradient(loss, model.trainable_variables)
    optim.apply_gradients(zip(grads, model.trainable_variables))

    if epoch % 250 == 0:
        y_train_pred = model(x_train)
        y_train_pred_fft = myFFT(y_train_pred)
        fig = plt.figure()
        plt.semilogy((y_train_pred_fft+1e-5)/train_size,'g*-')
        plt.semilogy((fft_train+1e-5)/train_size,'ro-')
        plt.xlabel('Freqence')
        plt.ylabel('|FFT|')
        plt.title('Train Epoch %d' % epoch)
        # plt.show()
        plt.savefig(fig_path+'trainfft'+str(epoch)+'.png',dpi=100)
        fig = plt.figure()
        plt.plot(x_train, (y_train_pred),'g*-')
        plt.plot(x_train, (y_train),'ro-')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Train Epoch %d' % epoch)
        # plt.show()
        plt.savefig(fig_path+'train'+str(epoch)+'.png',dpi=100)
        y_test_pred = model(x_test)
        y_test_pred_fft = myFFT(y_test_pred)
        fig = plt.figure()
        plt.semilogy((y_test_pred_fft+1e-5)/test_size,'g*-')
        plt.semilogy((fft_test+1e-5)/test_size,'ro-')
        plt.xlabel('Freqence')
        plt.ylabel('|FFT|')
        plt.title('Test Epoch %d' % epoch)
        # plt.show()
        plt.savefig(fig_path+'testfft'+str(epoch)+'.png',dpi=100)
        fig = plt.figure()
        plt.plot(x_test, (y_test_pred),'g*-')
        plt.plot(x_test, (y_test),'ro-')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Test Epoch %d' % epoch)
        # plt.show()
        plt.savefig(fig_path+'test'+str(epoch)+'.png',dpi=100)
