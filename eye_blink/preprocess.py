from helpers import *
import matplotlib.pyplot as plt
import os, glob, cv2, random
import seaborn as sns
import pandas as pd

base_path = 'D:\capstone'

X, y = read_csv(os.path.join(base_path, 'dataset.csv'))

#print(X.shape, y.shape)

# 내가 봤을때 왼쪽 눈
plt.figure(figsize=(12, 10))
for i in range(50):
    plt.subplot(10, 5, i+1)
    plt.axis('off')
    plt.imshow(X[i].reshape((26, 34)), cmap='gray')

#processing
n_total = len(X)
X_result = np.empty((n_total, 26, 34, 1))

for i, x in enumerate(X):
    img = x.reshape((26, 34, 1))

    X_result[i] = img

from sklearn.model_selection import train_test_split

x_train, x_val, y_train, y_val = train_test_split(X_result, y, test_size=0.1)

print(x_train.shape, y_train.shape)
print(x_val.shape, y_val.shape)
#
# np.save('x_train.npy', x_train)
# np.save('y_train.npy', y_train)
# np.save('x_val.npy', x_val)
# np.save('y_val.npy', y_val)

#분할된 데이터 시각화
fig, axs = plt.subplots(2, 1, figsize=(8, 6))
axs[0].set_title(str(y_train[0]))
axs[0].imshow(x_train[0].reshape((26, 34)), cmap='gray')
axs[1].set_title(str(y_val[5]))
axs[1].imshow(x_val[5].reshape((26, 34)), cmap='gray')
for ax in axs:
    ax.axis('off')
plt.show()

