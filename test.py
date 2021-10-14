from utils import *
from data_gen import *

# Set framework
sm.set_framework('tf.keras')
sm.framework()

# Instanciate the unet model to predict the localization of a tumor in an image
model = sm.Unet('resnet152', classes=3, input_shape=(256, 256, 3), activation='softmax')


#compile the model
opt = Adam(learning_rate=1E-5, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
model.compile(
            optimizer=opt,
            loss = 'categorical_crossentropy',
            metrics= [tf.keras.metrics.MeanIoU(num_classes=3), 
                    tf.keras.metrics.Precision(thresholds=0.5, class_id=1)]
            )

# Import the saved test dataframe
test_df = pd.read_csv("test_df.csv")

# Instanciate the test generator
test_gen = DataGenerator(test_df.indexes, test_df.T2reg_img_path, test_df.Mask_path, batch_size=1, n_classes=3, shuffle=True)

# Select a slice image associated to a tumor
test_im, test_mask = test_gen.__getitem__(6)

# Load the weights of the trained model
model.load_weights('weights/Fouad_model.h5')

# Predict the localization of the tumor from the image
test_results = model.predict(test_im)
plt.title("pred_mask")
plt.imshow(np.argmax(test_results[0,:,:,:], axis =-1))
plt.show()
plt.title("image")
original = np.reshape(test_im,(256,256,3))[:,:,0]
plt.imshow(original)
plt.show()
plt.title("real_mask")
plt.imshow(np.argmax(test_mask[0,:,:,:], axis =-1))
plt.show()