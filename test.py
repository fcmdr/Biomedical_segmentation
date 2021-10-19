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
test_gen = DataGenerator(test_df.indexes, test_df.T2reg_img_path, test_df.Mask_path, batch_size=1, augment=False, n_classes=3, shuffle=False)

# Select a slice image associated to a tumor
# INDEX = 10
# for i in range(INDEX):
#         test_im, test_mask = test_gen.__getitem__(i)
#         #visualizing prediction
#         original = test_im[0,:,:,:]
#         mask_pred = test_mask[0,:,:,:]
#         print(original)
#         original = cv2.cvtColor(original,cv2.COLOR_GRAY2RGB)
#         print(original.shape)
#         #cv2.imshow('Brain MRI',original)
#         # display blue image with overlay
#         dst = cv2.addWeighted(original,0.8,mask_pred,0.7,0)
#         #cv2.imshow('Brain MRI/Brain MRI tumor mask',dst)
#         img_arr = np.hstack((original, dst))
#         cv2.imshow('Brain MRI/Brain MRI tumor mask',img_arr)
#         cv2.waitKey(2000)
#         cv2.destroyAllWindows()

test_im, test_mask = test_gen.__getitem__(6)
test_mask = 255 * test_mask
test_mask[0,:,:,0] = 0
test_mask[0,:,:,2] = 0


# Create a plot to compare image and overlay (image and tumor mask superposed)
fig, ax = plt.subplots(5,2,figsize=(30,30))
# Set titles
ax[0][0].title.set_text("Image only")
ax[0][1].title.set_text("True mask (Brain MRI + tumor)")

for x in range(5):
    test_im, test_mask = test_gen.__getitem__(x)
    # Convert into integer
    test_im = test_im.astype(np.uint8)
    # Multiply by 255 to display the image
    test_im = 255 * test_im
    # Plot the image in grayscale using the 3 channels of the matrix
    ax[x][0].imshow(test_im[0,:,:,:])
    # Convert to integer
    test_mask = test_mask.astype(np.uint8)
    # Multiply by 255
    test_mask = 255 * test_mask
    # Set information from 1st and last channel to zero as the information about the tumor is contained into the 2nd channel (or the 2nd class)
    test_mask[0,:,:,0] = 0
    test_mask[0,:,:,2] = 0
    # Change order of the channel to display the mask in red color
    test_mask = test_mask[:,:,:,[1,0, 2]] 
    # Create the overlay (superposition of the brain image and the tumor mask)
    overlay = cv2.addWeighted(test_im[0,:,:,:], 0.5,test_mask[0,:,:,:],1.2,0)
    # Plot the overlay
    ax[x][1].imshow(overlay)

plt.show()
raise

#     test_results = model.predict(test_im)
#     print(f"test_results : {np.unique(test_results)}")

#     test_results = test_results.astype(np.uint8)
#     print(f"test_results : {np.unique(test_results)}")
#     plt.show()
#     raise

#     test_results = test_results * 255
#     test_results[0,:,:,0] = 0
#     test_results[0,:,:,2] = 0
    
#     test_results = test_results[:,:,:,[1,0, 2]] 
#     ax[x][2].imshow(test_results[0,:,:,:])

# #     ax[x][2].imshow(test_im[0,:,:,:])
# #     ax[x][2].imshow(test_mask[0,:,:,:], alpha=0.4)
# plt.show()
# raise


# # loop over the alpha transparency values
# for alpha in np.arange(0, 1.1, 0.1)[::-1]:
# 	# create two copies of the original image -- one for
# 	# the overlay and one for the final output image
# 	overlay = test_im.copy()
# 	output = test_im.copy()
# 	# draw a red rectangle surrounding Adrian in the image
# 	# along with the text "PyImageSearch" at the top-left
# 	# corner
#         cv2.addWeighted(overlay, alpha, output, 1 - alpha,0, output)
#         print("alpha={}, beta={}".format(alpha, 1 - alpha))
# 	cv2.imshow("Output", output)
# 	cv2.waitKey(1000)
# raise
# plt.show()
# raise

def display(display_list):
  plt.figure(figsize=(15, 15))

  title = ['Input Image', 'True Mask', 'Predicted Mask']

  for i in range(len(display_list)):
    plt.subplot(1, len(display_list), i+1)
    plt.title(title[i])
    plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
    plt.axis('off')
  plt.show()

for images, masks in test_gen:
        print("shape of images and masks are :")
        print(images.shape,masks.shape)
        raise
#   sample_image, sample_mask = images[0], masks[0]
#   display([sample_image, sample_mask])

raise
fig, axs = plt.subplots(1,2, figsize=(10,10), squeeze=False)

for i in range(10):
        test_im, test_mask = test_gen.__getitem__(i)
        print(test_mask.shape)
        #read mri images
        original = test_im[0,:,:,0]
        print(original.shape)
        #img = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
        #plt.imshow(original)
axs[count][0].imshow(original)
axs[count][0].title.set_text('Brain MRI')
plt.show()
raise

        # raise

        # plt.show()
        # #Save image
        # # Filename 
        # filename = 'overlay_image.jpg'
        # # Using cv2.imwrite() method 
        # # Saving the image 
        # #cv2 seems to save image in BGR format so use plt.imsave
        # plt.imsave(filename, img_masked.astype(np.uint8)) 
        # plt.imshow(img_masked.astype(np.uint8))

        # raise
# raise
#         #read original mask
#         mask = io.imread(df_pred.mask_path[i])
#         axs[count][1].imshow(mask)
#         axs[count][1].title.set_text('True mask')
        
#         #read predicted mask
#         pred = np.array(df_pred.predicted_mask[i]).squeeze().round()
#         axs[count][2].imshow(pred)
#         axs[count][2].title.set_text('Predicted mask')
        
#         #overlay original mask with MRI
#         img[mask==255] = (255,0,0)
#         axs[count][3].imshow(img)
#         axs[count][3].title.set_text('Brain MRI with true mask')
        
#         #overlay predicted mask and MRI
#         img_ = io.imread(df_pred.image_path[i])
#         img_ = cv2.cvtColor(img_, cv2.COLOR_BGR2RGB)
#         img_[pred==1] = (0,255,150)
#         axs[count][4].imshow(img_)
#         axs[count][4].title.set_text('Brain MRI with predicted mask')
        
#         count +=1
#     if (count==15):
#         break
# raise




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