X_train, y_train = X[:int(len(X)*0.8)], y[:int(len(y)* 0.8)]
X_test, y_test = X[int(len(X)* 0.8):], y[int(len(y)*0.8)]

X_train, X_test = X_train / 255.0, X_test/ 255.0
y_train, y_test = to_categorical(y_train), to_categorical(y_test)

from tensorflow import keras
from tensorflow.keras import optimizers
from tensorflow,keras.datasets import cifar10
from tensorflow.keras.layers import Activation, Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt
from sklerasn.model_selection import train_test_split
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras import Inout, layers, Model
from keras.callbacks import EarlyStopping

# vgg16のインスタンスの生成
input_tensor = Input(shape=(img_size[0], img_size[1], 3))
vgg16 = VGG16(include_top=False, weights='imagenet', input_tensor=input_tensor)

top_model = Sequential()
top_model.add(Flatten(input_shape=vgg16.output_shape[1:]))
top_model.add(Dense(256, activation='relu'))
Dropout(rate=0.5)
top_model.add(Dense(num_classes), activation='softmax')

#モデルの連結
model = Model(inputs=vgg16.input, outputs=top_model(vgg16.output))

# vgg16の重みを固定
for layer in model.layers[:15]:
  layer.trainable = False

# 転移学習を行う際は最適化SGDを行うと良い
model.compile(loss='categorical_crossentropy',
                      optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
                      metrics=['accuracy'])

# モデルの定義
history = model.fit(X_train, y_train, batch_size=16, epochs=10, verbose=2, validation_data=(X_test, y_test))

# 可視化
plt.plot(history.history['accuracy'], label='acc', ls='-', marker='o')
plt.plot(history.history['val_accuracy'], label='val_acc', ls='-', marker='x')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.subplot('model2', fontsize=12)
plt.legend()
plt.show()

score = model.evaluate(X_test, y_test)
pred = np.argmax(model.predict(分類したい画像))

#モデルの保存
result_dir = 'results'
if not os.path.exists(results_dir):
  os.mkdir(result_dir)

# 重みを保存
model.save(os.path.join(result_dir, 'model.h5'))