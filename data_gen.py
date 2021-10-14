from utils import *
import tensorflow as tf
import os,pydicom, shutil
import cv2
import imgaug as ia
from imgaug import augmenters as iaa

def getPixelDataFromDicom(filename):
        """Get pixel values from a dicom file""" 
        return pydicom.read_file(filename).pixel_array


class Dataset():
    def __init__(self,base_folder):
        """Define the "__init__ method" of the Dataset class which builds here all necessaries derived folder for images and masks from a base folder"""
        self.base_folder = base_folder
        self.dcm_files = os.path.join(base_folder,'data/dcm_files')
        self.jpg_folder_path = os.path.join(base_folder,'data/jpg_folder')
        self.image_folder_path = os.path.join(self.jpg_folder_path,'image')
        self.mask_folder_path = os.path.join(self.jpg_folder_path,'mask')

    def create(self):
        """Create folders for images and masks"""
        # Create list of folders 
        folders = [self.dcm_files,\
                   self.jpg_folder_path,\
                   self.image_folder_path,\
                   self.mask_folder_path
                  ]
        #for each folder build the folder if does not exist
        for folder in folders:
            os.makedirs(folder, exist_ok=True)

    def rename_files(self,lst):
        """Rename dcm files to include information about slice, patient id and modality
        the masks are put in a folder and the images in another folder"""
        for _path in lst: #Loop into path_files
            file_name = os.path.basename(_path) #Extract filename with os.path.basename function
            dir_name=os.path.dirname(_path)   #Extract name of directory for each path
            dir_name_sub = re.sub(r'^.*?PGBM', 'PGBM', dir_name)
            dir_name_cleaned = dir_name_sub.replace("\\","_")#In the filename replace "\\" by "_"

            print(dir_name_cleaned)
            raise


            dir_name_cleaned = dir_name_cleaned.split("_")
            #dir_name_cleaned = dir_name_cleaned[:-1]
            dir_name_cleaned = '_'.join(dir_name_cleaned)

            if "MaskTumor" in dir_name_sub:
                source=_path #Define the sources files (contains the path of the files we want the names to be modified)
                destination=(self.mask_folder_path+"\\"+dir_name_cleaned+"_"+file_name)#the name we want to give for each files
                shutil.copy(source,destination)#os.rename function rename the files defined in "source" into the name defined in "destination"
            else:
                source=_path #Define the sources files (contains the path of the files we want the names to be modified)
                destination=(self.image_folder_path+"\\"+dir_name_cleaned+"_"+file_name)#the name we want to give for each files
                shutil.copy(source,destination)#os.rename function rename the files defined in "source" into the name defined in "destination"


    def transform_dcm_im_png(self):  
        """Transform dcm files into png files and move them into image folder """  
        for n, file in enumerate(os.listdir(self.image_folder_path)):
            ds = pydicom.dcmread(os.path.join(self.image_folder_path,file))
            pixel_array_numpy = ds.pixel_array
            image_name = file.replace("dcm","png")
            # Save image in "png" format
            cv2.imwrite(os.path.join(self.image_folder_path, image_name), pixel_array_numpy)
        print("Images generated !")

    def transform_dcm_msk_png(self):  
        """Transform dcm files into png files and move them into mask folder """  
        for n, file in enumerate(os.listdir(self.mask_folder_path)):
            ds = pydicom.dcmread(os.path.join(self.mask_folder_path,file))
            pixel_array_numpy = ds.pixel_array
            image_name = file.replace("dcm","png")
            # Save image in "png" format
            cv2.imwrite(os.path.join(self.mask_folder_path, image_name), pixel_array_numpy)
        print("Masks generated !")
        
def process_each_patient(patient_id):
    """
    function which takes the patient id and return a dataframe with masks and images aligned as pairs
    """
    base_folder_brain = os.getcwd()

    # list of masks (can also be written : str("%03d" % i)) (Masks)

    # to remove
    #masks_list = glob2.glob('data\\PGBM-{0:03d}/*/*Mask*/*.dcm'.format(patient_id))

    Mask_files = glob2.glob(os.path.join(base_folder_brain,"data\\PGBM-{0:03d}\\*\\*Mask*\\*.dcm").format(patient_id))

    # list of images (can also be written : str("%03d" % i)) (T2 modality)
    # to remove
    #t2_imgs_list = glob2.glob('data\\PGBM-{0:03d}/*/*T2*/*.dcm'.format(patient_id))
    T1post_files = glob2.glob(os.path.join(base_folder_brain,"data\\PGBM-{0:03d}\\*\\*T1post*\\*.dcm".format(patient_id)))
    T1prereg_files = glob2.glob(os.path.join(base_folder_brain,"data\\PGBM-{0:03d}\\*\\*T1prereg*\\*.dcm".format(patient_id)))
    FLAIR_files = glob2.glob(os.path.join(base_folder_brain,"data\\PGBM-{0:03d}\\*\\*FLAIR*\\*.dcm".format(patient_id)))
    ADC_files = glob2.glob(os.path.join(base_folder_brain,"data\\PGBM-{0:03d}\\*\\*ADC*\\*.dcm".format(patient_id)))
    T2reg_files = glob2.glob(os.path.join(base_folder_brain,"data\\PGBM-{0:03d}\\*\\*T2reg*\\*.dcm".format(patient_id)))


    df_imgs_list_split = pd.DataFrame(zip(T1post_files,T1prereg_files,FLAIR_files,ADC_files,T2reg_files,Mask_files), columns = ["T1_post_img_path","T1prereg_img_path","FLAIR_img_path","ADC_img_path","T2reg_img_path", "Mask_path"])

            
    # s = "/data/PGBM-001/04-02-1992-NA-FH-HEADBrain Protocols-79896/"
    # pattern = "\\data\\(.*?)\\"
    masks_list_split = [mask.split("\\") for mask in  Mask_files]
    

    # Create column "Patient_id"
    for mask in masks_list_split:
        substring = mask[4]
        df_imgs_list_split['Patient_id'] = substring
    

    #  create a dataframe where the study id , image id , the patient id , slice id
    # df_patient_pair_sorted = df_imgs_list_split.merge(df_masks_list_split, on = ["patient_id", "study_id", "slice_id"]).sort_values(by=["patient_id",	
    #                                                                                                         "study_id",	
    #                                                                                                         "modality_id",
    #                                                                                                         "mask_id",
    #                                                                                                         "slice_id"]).reset_index(drop=True)
    # check that the number of lines == number of pairs (images, masks)
    # assert df_patient1_pair_sorted.shape[0] == total_pairs

    # aligned dataframe
    return df_imgs_list_split

# Oversample the tumors sample in the dataset
def oversample(df, patient_ID):
  """Oversample the minority class (tumors in our case).
  Inputs
  df : Is a dataframe containing images and corresponding masks paths, and patient_ID
  patient_ID : Identifier of the patients from '001' to '020'.
  returns : The concatenated dataframe containing images and corresonding tumor mask
  with the number of slices without tumor same as the number of slices with tumors."""

  df['Unique_values'] = [np.unique(getPixelDataFromDicom(mask)) for mask in df['Mask_path']]
  df['Tumor'] = [int(np.any(val)) for val in df['Unique_values']]
  df_patient = df[df['Patient_id'] == patient_ID]

  # select patient with tumor
  df_tumor = df[df['Tumor'] == 1]

  # Count number of tumors by Patient
  df_tumor.groupby(by = "Patient_id").agg("count")[["Tumor"]]

  # Extract index of slices with tumor for patient 1
  index_list_df_tumor = list(df_tumor[df_tumor["Patient_id"] == patient_ID].index)

  # Get nb of slice without tumor for each patient
  df_patient_no_tumor = df_patient[df_patient['Tumor'] == 0]

  # Oversample tumor n times (with n nb of slice without tumors)
  index_list_df_tumor_oversampled = np.random.choice(index_list_df_tumor, len(df_patient_no_tumor))

  # Apply the index list to the df dataframe
  df_tumor_patient = df.iloc[index_list_df_tumor_oversampled]

  # Reset the index and keep the Patient as a column
  df_tumor_patient_clean = df_tumor_patient.reset_index(drop = True)

  # For patient 1 select slices without tumor
  df_patient_no_tumor = df_patient[df_patient['Tumor'] == 0]

  # reset index before the concatenation with the dataset without tumor
  frames = [df_tumor_patient_clean, df_patient_no_tumor]
  df_tumor_patient_clean = df_tumor_patient_clean.reset_index()
  df_patient_no_tumor = df_patient_no_tumor.reset_index()
  result = pd.concat(frames)

  assert len(result[result['Tumor'] == 0]) == len(result[result['Tumor']==1])
  return result

# Version 1 (works) datagenerator without data augmentation
# class DataGenerator(tf.keras.utils.Sequence):
#     def __init__(self, list_IDs, img_path, mask_path, batch_size=1, dim=(256,256), n_channels=3,
#                 n_classes=3, shuffle=True):
#         self.dim = dim
#         self.batch_size = batch_size
#         self.img_path = img_path
#         self.mask_path = mask_path
#         self.list_IDs = list_IDs
#         self.n_channels = n_channels
#         self.n_classes = n_classes
#         self.shuffle = shuffle
#         self.on_epoch_end()

#     def __len__(self):
#         return len(self.list_IDs) // (self.batch_size)

#     def __getitem__(self, index):
#         index = self.index[index * self.batch_size:(index + 1) * self.batch_size]
#         list_IDs_temp = [self.index[k] for k in index]

#         X, y = self.__data_generation(list_IDs_temp)
#         return X, y

#     def on_epoch_end(self):
#         self.index = np.arange(len(self.list_IDs))
#         if self.shuffle == True:
#             np.random.shuffle(self.list_IDs)
    
#     def __data_generation(self, list_IDs_temp):
#         'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
#         # Initialization
#         X = np.empty((self.batch_size, *self.dim, self.n_channels))
#         y = np.empty((self.batch_size, *self.dim, self.n_channels))

#         # Generate data
#         for i, ID in enumerate(list_IDs_temp):
#             # replace double by single backslash
#             self.img_path.iloc[ID] = self.img_path.iloc[ID].replace("\\\\","\\")
            
#             # Store sample
#             ds_image = pydicom.dcmread(self.img_path.iloc[ID].replace("\\\\","\\"))
#             image_array = ds_image.pixel_array

#             # Resize the image array
#             images_resized = cv2.resize(image_array, 
#                                         (self.dim), 
#                                         interpolation = cv2.INTER_LINEAR)

#             # Divide the images by 255 to get values between 0 and 1
#             images_rescaled = [x/255 for x in images_resized]
#             images_rescaled = cv2.merge([images_resized,images_resized,images_resized])

#             # Modify shape adding 1 channel to match the neural network input size
#             #images_rescaled = np.expand_dims(images_rescaled, axis = -1)

#             # Add another dimension for the channel 
#             images_batched = np.expand_dims(images_rescaled, axis = 0)

#             X[i,] = images_batched

#             # Store class
#             ds_mask = pydicom.dcmread(self.mask_path[ID])

#             mask_array = ds_mask.pixel_array

#             # Resize the image array
#             masks_resized = cv2.resize(mask_array, 
#                                         (self.dim), 
#                                         interpolation = cv2.INTER_LINEAR)

#             # Convert the mask into dimension (x, y, n_classes), to get binary value for each class by channel
#             masks_3d = tf.keras.utils.to_categorical(masks_resized, num_classes = self.n_channels)

#             # Add another dimension for the channel 
#             masks_3d = np.expand_dims(masks_3d, axis = 0)
#             y[i,] = masks_3d
            
            
        
#         return X, y
        
# Version 2 (works) datagenerator with data augmentation
class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, list_IDs, img_path, mask_path, batch_size=1, dim=(256,256), n_channels=3,
                n_classes=3, augment=True, shuffle=True):
        self.dim = dim
        self.batch_size = batch_size
        self.img_path = img_path
        self.mask_path = mask_path
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.augment = augment
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return len(self.list_IDs) // (self.batch_size)

    def __getitem__(self, index):
        index = self.index[index * self.batch_size:(index + 1) * self.batch_size]
        list_IDs_temp = [self.index[k] for k in index]

        # Apply pre-processing (including data augmentation)
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        self.index = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.list_IDs)
    
    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size, *self.dim, self.n_channels))

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # replace double by single backslash
            self.img_path.iloc[ID] = self.img_path.iloc[ID].replace("\\\\","\\")

            #print(self.img_path.iloc[ID])
            
            # Store sample
            ds_image = pydicom.dcmread(self.img_path.iloc[ID].replace("\\\\","\\"))
            image_array = ds_image.pixel_array

            # Resize the image array
            images_resized = cv2.resize(image_array, 
                                        (self.dim), 
                                        interpolation = cv2.INTER_LINEAR)

            # Divide the images by 255 to get values between 0 and 1
            #images_rescaled = [x/255 for x in images_resized]
            pixels = np.asarray(images_resized)
            # convert from integers to floats
            pixels = pixels.astype('float32')
            # calculate global mean and standard deviation
            mean, std = pixels.mean(), pixels.std()
            # print('Mean: %.3f, Standard Deviation: %.3f' % (mean, std))
            # global standardization of pixels
            pixels = (pixels - mean) / std
            # clip pixel values to [-1,1]
            pixels = np.clip(pixels, -1.0, 1.0)
            # shift from [-1,1] to [0,1] with 0.5 mean
            pixels = (pixels + 1.0) / 2.0
            # confirm it had the desired effect
            mean, std = pixels.mean(), pixels.std()
            # print('Mean: %.3f, Standard Deviation: %.3f' % (mean, std))
            # print('Min: %.3f, Max: %.3f' % (pixels.min(), pixels.max()))
            images_rescaled = cv2.merge([pixels,pixels,pixels])

            if self.augment:
                images_rescaled = self.__data_augmentation(images_rescaled)

            # Modify shape adding 1 channel to match the neural network input size
            #images_rescaled = np.expand_dims(images_rescaled, axis = -1)

            # Add another dimension for the channel 
            images_batched = np.expand_dims(images_rescaled, axis = 0)

            X[i,] = images_batched

            # Store class
            ds_mask = pydicom.dcmread(self.mask_path[ID])

            mask_array = ds_mask.pixel_array

            # Resize the image array
            masks_resized = cv2.resize(mask_array, 
                                        (self.dim), 
                                        interpolation = cv2.INTER_LINEAR)

            # Convert the mask into dimension (x, y, n_classes), to get binary value for each class by channel
            masks_3d = tf.keras.utils.to_categorical(masks_resized, num_classes = self.n_channels)

            # Add another dimension for the channel 
            masks_3d = np.expand_dims(masks_3d, axis = 0)
            y[i,] = masks_3d
            
        return X, y
    
    def __data_augmentation(self, img):
        ''' function for apply some data augmentation '''
        img = tf.keras.preprocessing.image.apply_brightness_shift(img, 0.2)
        img = tf.image.random_saturation(img, 0.6, 1.6)
        img = tf.image.random_brightness(img,0.2)
        img = tf.image.stateless_random_contrast(img, 0.2, 0.5, seed=(1, 2))
        return img

# Version 3 Datagenerator with data augmentation and all modalities
# class DataGenerator(tf.keras.utils.Sequence):
#     def __init__(self, list_IDs,T1prereg_img_path, T1_post_img_path, FLAIR_img_path, ADC_img_path,	T2reg_img_path, mask_path, 
#                 batch_size=1, dim=(256,256), n_modalities=5 ,n_classes=5, augment=True, shuffle=True):
#         self.dim = dim
#         self.batch_size = batch_size
#         self.T1prereg_img_path = T1prereg_img_path
#         self.T1_post_img_path = T1_post_img_path
#         self.FLAIR_img_path = FLAIR_img_path
#         self.ADC_img_path = ADC_img_path
#         self.T2reg_img_path = T2reg_img_path
#         self.Mask_path = mask_path
#         self.list_IDs = list_IDs
#         self.n_modalities = n_modalities
#         self.n_classes = n_classes
#         self.augment = augment
#         self.shuffle = shuffle
#         self.on_epoch_end()

#     def __len__(self):
#         return len(self.list_IDs) // (self.batch_size)

#     def __getitem__(self, index):
#         index = self.index[index * self.batch_size:(index + 1) * self.batch_size]
#         list_IDs_temp = [self.index[k] for k in index]

#         # Apply pre-processing (including data augmentation)
#         X, y = self.__data_generation(list_IDs_temp)

#         return X, y

#     def on_epoch_end(self):
#         self.index = np.arange(len(self.list_IDs))
#         if self.shuffle == True:
#             np.random.shuffle(self.list_IDs)
    
#     def preprocess_image(self, list_IDs_temp, img_path):
#         # Initialization
#         X = np.empty((self.batch_size, *self.dim))

#         # Generate data
#         for i, ID in enumerate(list_IDs_temp):
#             # replace double by single backslash
#             img_path.iloc[ID] = img_path.iloc[ID].replace("\\\\","\\")
            
#             # Store sample
#             ds_image = pydicom.dcmread(img_path.iloc[ID].replace("\\\\","\\"))
#             image_array = ds_image.pixel_array

#             # Resize the image array
#             images_resized = cv2.resize(image_array, 
#                                         self.dim, 
#                                         interpolation = cv2.INTER_LINEAR)

#             # Divide the images by 255 to get values between 0 and 1
#             images_rescaled = images_resized / 255 

#             #images_rescaled = cv2.merge([images_resized,images_resized,images_resized])

#             # if self.augment:
#             #     images_rescaled = self.__data_augmentation(images_rescaled)

#             # Modify shape adding 1 channel to match the neural network input size
#             #images_rescaled = np.expand_dims(images_rescaled, axis = -1)

#             # Add another dimension for the channel 
#             #images_batched = np.expand_dims(images_rescaled, axis = 0)

#             X[i,] = images_rescaled

#         return X

#     def __data_generation(self, list_IDs_temp):
#         'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
#         # Process images
#         X_img = np.empty((self.batch_size, *self.dim, 5))


#         modalities = [self.T1_post_img_path, self.T1prereg_img_path, self.FLAIR_img_path, self.ADC_img_path, self.T2reg_img_path]
#         for i, modality in enumerate(modalities):
#             img = self.preprocess_image(list_IDs_temp, modality)
#             img = np.expand_dims(img, axis = 0)
#             #X_img.append(np.array(img))
#             X_img[:,:,:,i] = img
#         X = X_img
        

#         # Process masks
#         y = np.empty((self.batch_size, *self.dim, self.n_modalities))
#         for i, ID in enumerate(list_IDs_temp):
#             # Store class
#             ds_mask = pydicom.dcmread(self.Mask_path[ID])

#             mask_array = ds_mask.pixel_array

#             # Resize the image array
#             masks_resized = cv2.resize(mask_array, 
#                                         (self.dim), 
#                                         interpolation = cv2.INTER_LINEAR)

#             # Convert the mask into dimension (x, y, n_classes), to get binary value for each class by channel
#             masks_3d = tf.keras.utils.to_categorical(masks_resized, num_classes = self.n_modalities)

#             # Add another dimension for the channel 
#             #masks_3d = np.expand_dims(masks_3d, axis = 0)
#             y[i,] = masks_3d
            
#         return X, y
    
#     def __data_augmentation(self, img):
#         ''' function for apply some data augmentation '''
#         img = tf.keras.preprocessing.image.apply_brightness_shift(img, 0.2)
#         img = tf.image.random_saturation(img, 0.6, 1.6)
#         img = tf.image.random_brightness(img,0.2)
#         img = tf.image.stateless_random_contrast(img, 0.2, 0.5, seed=(1, 2))
#         return img