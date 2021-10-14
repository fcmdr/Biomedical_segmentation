from utils import *
from data_gen import *
from models import *
    
if __name__ == '__main__':
    # Define "base_folder_brain" in which the folder data is located
    base_folder_brain = os.getcwd()

    # Files = glob2.glob(os.path.join(base_folder_brain,"data/*/*/*/*.dcm"))
    # Mask_files = glob2.glob(os.path.join(base_folder_brain,"data/*/*/*Mask*/*.dcm"))

    # img_files = set(Files) - set(Mask_files)
    # print(f"len of total nb of files is: {len(Files)}")
    # print(f"len of total nb of images is: {len(img_files)}")

    # print(f"len of total nb of masks is: {len(Mask_files)}")
    

    
    # modalities = ['T1post','dT1','T1prereg','FLAIR','ADC','sRCBVreg','nRCBVreg','nCBFreg','T2reg']
    # for modality in modalities:
    #     modality = glob2.glob(os.path.join(base_folder_brain,f"data/*/*/*{modality}*/*.dcm"))
    # print(T1post)
    # raise


    T1post_files = glob2.glob(os.path.join(base_folder_brain,"data/*/*/*T1post*/*.dcm"))
    T1prereg_files = glob2.glob(os.path.join(base_folder_brain,"data/*/*/*T1prereg*/*.dcm"))
    FLAIR_files = glob2.glob(os.path.join(base_folder_brain,"data/*/*/*FLAIR*/*.dcm"))
    ADC_files = glob2.glob(os.path.join(base_folder_brain,"data/*/*/*ADC*/*.dcm"))
    T2reg_files = glob2.glob(os.path.join(base_folder_brain,"data/*/*/*T2reg*/*.dcm"))
    Mask_files = glob2.glob(os.path.join(base_folder_brain,"data/*/*/*Mask*/*.dcm"))

    # df_files = pd.DataFrame(list(zip(T1post_files,T1prereg_files,FLAIR_files,ADC_files,T2reg_files,Mask_files)))

    # Sanity check (check that the lists are not empty)
    #assert len(T2_files)!=0 or len(Mask_files)!=0, "check your path !" 

    
    # Instanciate the class Dataset
    fouad_dataset = Dataset(base_folder_brain)
    # # Create the needed folders
    fouad_dataset.create()

    # Create a list of list for every dcm files per modality
    modalities = [T1post_files, T1prereg_files, FLAIR_files,ADC_files, T2reg_files] 

    # # Rename the files and put them in the image or the mask folder
    # for modality in modalities:
    #      fouad_dataset.rename_files(modality)
    
    # fouad_dataset.rename_files(Mask_files)
    

    # Transform the dcm files into png files
    # fouad_dataset.transform_dcm_im_png()
    # fouad_dataset.transform_dcm_msk_png()



    ### Dataframe creation 
   
    # # empty list where we will append/put patients dataframes
    # patients_df = []
    # n_patients = 20

    # # Process all files for every patient to create the dataframe containing the differents paths
    # for i in range(1, n_patients + 1,1):
    #     patients_df.append(process_each_patient(i))
    # # dataframe concatenated with images-masks pairs
    # df_raw = pd.concat(patients_df, axis=0).reset_index(drop=True)


    # # Oversample dataframe
    # appended_data = []
    # # Create id list for all patients
    # list_patients_id = np.unique(df_raw['Patient_id'])
    # for patient in tqdm(list_patients_id):
    #     oversample_data = oversample(df = df_raw, patient_ID = patient)
    #     appended_data.append(oversample_data)
    # appended_data = pd.concat(appended_data)
    # appended_data['indexes'] = np.arange(len(appended_data))

    # ########### Train-test split #############

    # # Split the dataframe into train, validation and test using the concatenated dataframe
    # train_df, val_df, test_df = split_dataframe(appended_data, 0.8, 0.9) # Split the dataset randomly in train, validation and test

    # train_df.to_csv("train_df.csv")
    # val_df.to_csv("val_df.csv")
    # test_df.to_csv("test_df.csv")

    train_df = pd.read_csv("train_df.csv")
    val_df = pd.read_csv("val_df.csv")
    test_df = pd.read_csv("test_df.csv")

    ############################################################ Model ###############################################################################################

    # input_im_dcm_paths = glob2.glob(os.path.join(base_folder_brain,"data\\jpg_folder\\image\\","*.dcm"))
    # target_dcm_paths = glob2.glob(os.path.join(base_folder_brain, "data\\jpg_folder\\mask\\","*.dcm"))

    # Instanciate the parameters of the custom generator
    # Parameters of the custom generator

    train_gen = DataGenerator(train_df.indexes,	train_df.T2reg_img_path,
                                train_df.Mask_path, batch_size=1, augment=False, shuffle=True)



    val_gen = DataGenerator(val_df.indexes, val_df.T2reg_img_path,
                                val_df.Mask_path, batch_size=1, augment=False, shuffle=True)
    
    #1st working version
    # Define train and validation generators
    # train_gen = DataGenerator(train_df.indexes, train_df.img_path, train_df.mask_path, batch_size=1, n_classes=3, shuffle=True)
    # val_gen = DataGenerator(val_df.indexes, val_df.img_path, val_df.mask_path, batch_size=1, n_classes=3, shuffle=True)
        
    # Free memory before training the model
    tf.keras.backend.clear_session()
    sm.set_framework('tf.keras')

    # Load the unet model with weights associated to the resnet dataset
    model = sm.Unet('resnet152', classes=3, input_shape=(256, 256, 3), activation='softmax')
    
    opt = Adam(learning_rate=1E-5, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    
    # Define metrics
    METRICS = [
        MeanIoU(num_classes=3, name='mean_iou'), 
        Precision(thresholds=0.5, class_id=1, name='precision')
    ]
    #compile the model
    model.compile(
              optimizer=opt,
              loss=dice_coef_loss,
              metrics=METRICS)

    # Training of the model

    #for EPOCH in range(2):
        
    results = model.fit(train_gen, 
                        epochs = 10, 
                        validation_data = val_gen) 

                                # callbacks = callbacks_list)
    #Save the final model for each epoch
    model.save_weights(os.path.join('weights/','Fouad_model.h5'))
    # print(results.history.keys())

    # print(results.history.columns)
    
    # dict_keys(['loss', 'mean_io_u', 'precision', 'val_loss', 'val_mean_io_u', 'val_precision'])
    plt.plot(results.history['loss'], label='Categorical cross-entropy (training data)')
    plt.plot(results.history['val_loss'], label='Categorical cross-entropy (validation data)')
    plt.title('Categorical cross-entropy for Brain-tumor-segmentation')
    plt.ylabel('Categorical cross-entropy value')
    plt.xlabel('No. epoch')
    plt.legend(loc="upper left")
    plt.show()


