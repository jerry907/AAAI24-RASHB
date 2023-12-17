import numpy as np
import tensorflow_datasets as tfds

from loading import to_csv

ds_list = [tfds.load("mnist", split=name, as_supervised=True) for name in ["train", "test"]]

data = []

for ds in ds_list:
    for image, label in tfds.as_numpy(ds):
        im = np.array([image[i].flatten() for i in range(28)]).flatten()
        im = np.insert(im, 0, label)
        data.append(im)

data = np.array(data)

to_csv(data, filename="datasets/mnist_train.csv")