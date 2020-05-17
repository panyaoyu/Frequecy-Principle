import numpy as np
from matplotlib import pyplot as plt
from model import myModel
import os
import warnings
import random
from PIL import Image
import tensorflow as tf

warnings.filterwarnings("ignore")

img_path = './trainimg/face.png'
dest_path = './picture/'
model_path = './model/'

if not os.path.exists(dest_path):
    os.makedirs(dest_path)

if not os.path.exists(model_path):
    os.makedirs(model_path)

img = Image.open(img_path)
img.show()
img = np.array(img, dtype=np.float64)
mean = img.mean()
img -= mean
maxp = img.max()
img /= maxp
# print(img)

epochs = 100000
lr = 2e-5
MSE = tf.keras.losses.MeanSquaredError()
optim = tf.keras.optimizers.Adam(learning_rate=lr)
batch_size = 1024

# plt.imshow(img,plt.cm.gray)
# plt.show()

x_train = []
y_train = []
for i in range(len(img)):
    for j in range(0, len(img[0]), 2):
        x_train.append([i, j])
        y_train.append(img[i][j])
x_train = np.array(x_train)
y_train = np.array(y_train)

x_test = []
y_test = []
for i in range(len(img)):
    for j in range(len(img[0])):
        x_test.append([i, j])
        y_test.append(img[i][j])
x_test = np.array(x_test)
y_test = np.array(y_test)
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)
train_iter = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size)
test_iter = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)

model = myModel()
for param in model.trainable_variables:
    param = tf.random.truncated_normal(param.shape, mean=0, stddev=0.08)
train_loss, test_loss = [], []
for epoch in range(epochs):
    loss_sum_train = 0
    for X, y in train_iter:
        with tf.GradientTape() as tape:
            y_pred = model(X)
            loss = MSE(y_pred, y)
        grads = tape.gradient(loss, model.trainable_variables)
        optim.apply_gradients(zip(grads, model.trainable_variables))
        loss_sum_train += MSE(model(X), y)
    loss_sum_test = 0
    for X, y in test_iter:
        loss_sum_test += MSE(model(X), y)
    train_loss.append(loss_sum_train.numpy())
    test_loss.append(loss_sum_test.numpy())
    if epoch % 250 == 0:
        print("epoch %d, train_loss %lf, test_loss %lf" % (epoch, loss_sum_train, loss_sum_test))
        im = model(x_test).numpy()
        im = im * maxp + mean
        im = im.reshape(img.shape)
        plt.imshow(im, cmap=plt.cm.gray)
        plt.savefig(dest_path + '%d.png' % epoch)
        model.save_weights(model_path + '%d.h5' % epoch)

fig = plt.figure()
plt.plot(train_loss, label='train')
plt.plot(test_loss, label='test')
plt.legend()
plt.xlabel('epochs')
plt.ylabel('loss')
plt.savefig(dest_path + 'loss.png')
train_loss, test_loss = np.array(train_loss), np.array(test_loss)
np.save("train_loss.npy", train_loss)
np.save("test_loss.npy", test_loss)
