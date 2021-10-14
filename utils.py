import re
import random
import numpy as np # linear algebra
import pandas as pd # data processing
import glob2 
import imageio
import matplotlib.pyplot as plt
import data_gen
import math
import matplotlib.pyplot as plt
import pydicom
import random
from pydicom.data import get_testdata_files
#!pip install pydicom
import pydicom as dicom
import cv2
import matplotlib.pyplot as plt
import segmentation_models as sm
from segmentation_models.utils import set_trainable
from tqdm import tqdm
import imageio
import os
import shutil
import scipy.ndimage as ndi 
import skimage.io as io
import skimage.transform as trans
import tensorflow
from tensorflow.keras.utils import to_categorical ,Sequence
from tensorflow.keras.metrics import Precision,Recall,AUC, MeanIoU
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras import optimizers 
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras import initializers
from tensorflow.keras.preprocessing import image as image_utils
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler,EarlyStopping,TensorBoard
from tensorflow.keras import backend as keras
from tensorflow.keras import backend as K
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import imgaug as ia
from imgaug import augmenters as iaa

def getPixelDataFromDicom(filename):
        """Get pixel values from a dicom file""" 
        return pydicom.read_file(filename).pixel_array

def split_dataframe(dataframe,train_prop,one_minus_test_prop):
        """Split randomly the dataframe in which images and masks paths are keeped into train,validation and test"""
        train, validate, test = np.split(dataframe.sample(frac=1,random_state = 2), [int(train_prop*len(dataframe)), int(one_minus_test_prop*len(dataframe))])
        assert len(dataframe) == len(train) + len(validate) + len(test)
        return train, validate, test
