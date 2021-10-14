from utils import *
from data_gen import *
from models import *
def write_image(path, img):
    # img = img*(2**16-1)
    # img = img.astype(np.uint16)
    # img = img.astype(np.uint8)
    img = cv2.convertScaleAbs(img, alpha=(255.0))
    cv2.imwrite(path, img)

# Set framework
sm.set_framework('tf.keras')
sm.framework()

# Instanciate the unet model to predict the localization of a tumor in an image
model = sm.Unet('resnet152', classes=3, input_shape=(256, 256, 3), activation='softmax')


#compile the model
opt = Adam(learning_rate=1E-5)
model.compile(
            optimizer=opt,
            loss = dice_coef_loss,
            metrics= [tf.keras.metrics.MeanIoU(num_classes=3), 
                    tf.keras.metrics.Precision(thresholds=0.5, class_id=1)]
            )

# Import the saved test dataframe
test_df = pd.read_csv("test_df.csv")


# Instanciate the test generator
test_gen = DataGenerator(test_df.indexes, test_df.FLAIR_img_path, test_df.Mask_path, batch_size=1, augment=False, n_classes=3, shuffle=False)

# Select a slice image associated to a tumor
INDEX = 10
test_im, test_mask = test_gen.__getitem__(INDEX)
print(test_im.shape)
print(test_mask.shape)

# Load the weights of the trained model
model.load_weights('weights/Fouad_model.h5')

# Predict the localization of the tumor from the image
print(test_im.shape)

print(test_mask.dtype)
test_results = model.predict(test_im)
plt.title(f"pred_mask_{INDEX}")
plt.imshow(np.argmax(test_results[0,:,:,:], axis =-1))
write_image(f"pred_mask_{INDEX}.png", test_results[0,:,:,:])
plt.show()
plt.title(f"image{INDEX}")
original = np.reshape(test_im,(256,256,3))[:,:,0]
write_image(f"image_{INDEX}.jpg", original)
plt.imshow(original)
plt.show()
plt.title(f"real_mask_{INDEX}")
plt.imshow(np.argmax(test_mask[0,:,:,:], axis =-1))
write_image(f"real_mask_{INDEX}.png", np.argmax(test_mask[0,:,:,:], axis =-1))
plt.show()