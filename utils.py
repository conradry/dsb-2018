import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
import pandas as pd
import keras
from keras import backend as K
from keras.utils.data_utils import get_file
from keras.utils import np_utils
from keras.utils.np_utils import to_categorical
from keras.models import Sequential, Model
from keras.layers import Input, Embedding, Reshape, merge, LSTM, Bidirectional, add, concatenate
from keras.layers import TimeDistributed, Activation, SimpleRNN, GRU
from keras.layers.core import Flatten, Dense, Dropout, Lambda
from keras.regularizers import l2
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD, RMSprop, Adam
#from keras.utils.layer_utils import layer_from_config
from keras.metrics import categorical_crossentropy, categorical_accuracy
from keras.layers.convolutional import *
from keras.preprocessing import image, sequence
from keras.preprocessing.text import Tokenizer
from scipy.stats import pearsonr
from sklearn.metrics import confusion_matrix
from keras.layers import GlobalMaxPooling2D
from keras.layers import Permute
from keras.regularizers import *
from keras.preprocessing.image import ImageDataGenerator
import bcolz
from shutil import copyfile
from mean_average_precision import *

K.set_image_data_format('channels_last')

path = os.curdir + '/data/'
trn_path = path + 'train/'
tst_path = path + 'test1/'

if os.path.exists(path+'models/')==False:
    os.mkdir(path+'models/')
if os.path.exists(path+'results/')==False:
    os.mkdir(path+'results/')


def save_array(fname, arr):
    c=bcolz.carray(arr, rootdir=fname, mode='w')
    c.flush()

def load_array(fname):
    return bcolz.open(fname)[:]

#this will fill the nan angle values with a linear regression value
def angle_imputer(angle_data, band):
    #mask = np.isnan(angle_data)
    xna = np.where(np.isnan(angle_data)==True)[0]
    mins = np.array([min(train.band.iloc[i]) for i in xna])
    fit = np.array([abs(mins[i]*(-.80926)+9.346) for i in range(len(mins))])
    
    return np.put(angle_data, xna, fit)

def lr_anneal(model, x, y, val_x, val_y, lr, nb_epoch, verbose):
    model.compile(optimizer=Adam(lr=lr), loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(x, y, batch_size=32, nb_epoch=nb_epoch, verbose=verbose, 
              validation_data=(val_x, val_y))

def auto_anneal():
    lr_anneal(0.1, 1, 0)
    lr_anneal(0.01, 3, 0)
    lr_anneal(1e-3, 5, 0)
    lr_anneal(1e-4, 8, 0)
    lr_anneal(1e-5, 14, 0)
    return lr_anneal(1e-5, 1, 1)

#Make a new glob and shuf array with different ordering to get files for the sample training directory
def sample(src, dst, size):
    """Move files to sample directory
    # Arguments
        src: location of directory files that you're sampling, need trailing slash
        dst: location to move files, create dst if it doesn't exist, need trailing slash
        size: size of sample (number of files moved)
    """
    g = glob(src + '*.png')
    shuf = np.random.permutation(g)
    
    #check for destination, create it if is doesn't exist
    if os.path.exists(dst)==False:
        os.mkdir(dst)
    
    for i in range(size): 
        copyfile(shuf[i], dst + shuf[i].split('/')[-1])
        
def plots(ims, figsize=(12,6), rows=1, interp=False, titles=None):
    if type(ims[0]) is np.ndarray:
        ims = np.array(ims).astype(np.uint8)
        if (ims.shape[-1] != 3):
            ims = ims.transpose((0,2,3,1))
    f = plt.figure(figsize=figsize)
    cols = len(ims)//rows if len(ims) % 2 == 0 else len(ims)//rows + 1
    for i in range(len(ims)):
        sp = f.add_subplot(rows, cols, i+1)
        sp.axis('Off')
        if titles is not None:
            sp.set_title(titles[i], fontsize=16)
        plt.imshow(ims[i], interpolation=None if interp else 'none')
        
def plot_images(images, image_rows, image_cols):
    f, axarr = plt.subplots(image_rows, image_cols, figsize=(16, image_rows*2))
    for row in range(image_rows):
        for col in range(image_cols):
            image = images[row*image_cols+col]
            ax = axarr[row, col]
            ax.axis('off')
            ax.imshow(image)
            
def make_data_array(path, data_ids, get_masks=False):
    imgs=[]
    msks=[]
    img_sizes=[]
    for id_ in data_ids:
        img = pil_image.open(path+id_+'/images/'+id_+'.png')
        img = img.convert(mode='RGB')
        img_sizes.append(img.size)
        #height, width = img.size
        img = img.resize((256,256))
        img = np.asarray(img)
        imgs.append(img)
        
        if get_masks:
            msk = imread_collection(path+id_+'/masks/*.png').concatenate()
            num_masks = msk.shape[0]
            #essentially makes an empty image that's all black
            labels = np.zeros(msk.shape[1:], np.uint16)
            #for each pixel in a mask that is > 0 it adds that pixel to black image, 
            #color is incremented by 1 to discern individual cells
            for idx in range(num_masks):
                labels[msk[idx] > 0] = idx + 1
            labels = pil_image.fromarray(labels)
            labels = labels.resize((256,256))
            msk = np.expand_dims(np.asarray(labels), axis=2)
            msks.append(msk)
    if get_masks:
        return imgs, msks, img_sizes
    else:
        return imgs, img_sizes


get_ipython().magic(u'matplotlib inline')

#may cause memory allocation errors when training models
#from vis.visualization import visualize_saliency, visualize_activation

