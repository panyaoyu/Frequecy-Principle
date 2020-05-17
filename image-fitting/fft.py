# import tensorflow as tf
from model import myModel
from matplotlib import pyplot as plt
import numpy as np
import os
from PIL import Image

img_path = './trainimg/face.png'

model_path = './model/'
fft_path = './fft/'

if not os.path.exists(fft_path):
    os.mkdir(fft_path)

posx = 80

img = Image.open(img_path)
# img.show()
img = np.array(img, dtype=np.float64)
mean = img.mean()
img -= mean
maxp = img.max()
img /= maxp

def myFFT(x, freq_len=40):
    x = x.reshape(-1,1)
    fft = np.fft.fft2(x)
    return np.absolute(fft[range(freq_len)])

x_test = []
y_test = []
for i in range(len(img)):
    for j in range(len(img[0])):
        x_test.append([i, j])
        y_test.append(img[i][j])
x_test = np.array(x_test)
y_test = np.array(y_test)

data = x_test[posx*img.shape[1]:(posx+1)*img.shape[1]]
y_true = y_test[posx*img.shape[1]:(posx+1)*img.shape[1]]

y_true_fft = myFFT(y_true)

# plt.plot(y_true_fft, 'ro-')

model = myModel()
model(x_test)
for epoch in range(0, 100000, 250):
    model.load_weights(model_path+'%d.h5' % epoch)
    y = model(data).numpy()
    fft_data = myFFT(y)
    fig = plt.figure()
    plt.plot(fft_data/img.shape[1], 'g*-', label='Fit')
    plt.plot(y_true_fft/img.shape[1], 'ro-', label='True')
    plt.legend()
    plt.savefig(fft_path+'%d.png' % epoch)
