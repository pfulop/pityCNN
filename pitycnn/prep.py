import pandas as pd
import numpy as np
from os import path

def prepare_data(data_path):
    #     DATA_HOME = '/Users/fulop/Downloads/dogbreed/'
    #     DATA_HOME = '/run/media/backman/yay/dogbreed/'
    CSV = data_path + 'labels.csv'
    TRAIN_PATH = data_path + 'train/'
    VALID_PATH = data_path + 'valid/'
    dfile = pd.read_csv(CSV)
    dfile['breed'] = pd.Categorical(dfile['breed'])
    n_classes = len(dfile.breed.unique())
    dfile['breed'] = dfile.breed.cat.codes
    dfile['type'] = dfile.apply(
        lambda f: 'train' if path.isfile(TRAIN_PATH + f['id'] + '.jpg') else 'valid' if path.isfile(
            VALID_PATH + f['id'] + '.jpg') else 'none', axis=1)
    valid = dfile[dfile['type'] == 'valid']
    train = dfile[dfile['type'] == 'train']
    train_images = train['id'].map(lambda name: "{}{}.jpg".format(TRAIN_PATH, name)).values
    valid_images = valid['id'].map(lambda name: "{}{}.jpg".format(VALID_PATH, name)).values
    train_labels = train['breed'].values.astype(np.int32)
    valid_labels = valid['breed'].values.astype(np.int32)

    return train_images, train_labels, valid_images, valid_labels, n_classes
