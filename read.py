import os
path_datasets = 'flowers_datasets'
flowers_data = os

num_classes = len(labels)
image_size = 64, 64
all_images = []
all_labels = []

# 各フォルダから画像をリサイズしてフォルダごとのラベルに分ける
for label, flower in enumerate(flowers_data):
  for file in os.listdir(os.path.join(path_datasets, flowers)):
    if file.endswith("jpg"):
      # 画像のリサイズ
      img = cv2.omread(os.path.join(path_datasets, flowers, file))
      im = cv2.resize(img, img_size)
      all_images.append(im)

      #フォルダのindexをラベルとして使う
      all_labels.append(label)
    else:
      continue

# データをnumpy配列に変換する
import numpy as np
X = np.array(all_iamges)
y = np.array(all_labels)

# データをシャッフル
rand_index = np.ramdom.permutation(np.arange(len(X)))
X = X[rand_index]
y = y[rand_index]