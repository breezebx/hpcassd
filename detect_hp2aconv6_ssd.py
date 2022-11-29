import os
import time
import numpy as np
from math import ceil
from matplotlib import pyplot as plt
from keras import backend as K
from keras.optimizers import Adam
from models.keras_hp2aconv6_ssd import hp2aconv6_ssd
from keras_loss_function.keras_ssd_loss import SSDLoss

from data_generator.object_detection_2d_data_generator import DataGenerator
from data_generator.object_detection_2d_misc_utils import apply_inverse_transforms

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # CPU
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # GPU

dsfolder, dsname = './datasets/PCBData/', 'test'
showimg, saveimg, isprint, savetxt = True, True, True, True
# showimg, saveimg, isprint, savetxt = False, False, False, False
batch_size = 10
# Set a few configuration parameters.
img_height = 512
img_width = 512
img_channels = 2
n_classes = 6  # Number of positive classes

# Set this to your preference (maybe `None`). The current settings transform the input pixel values to the interval `[-1,1]`.
intensity_mean = 127.5
# Set this to your preference (maybe `None`). The current settings transform the input pixel values to the interval `[-1,1]`.
intensity_range = 127.5
# An explicit list of anchor box scaling factors. If this is passed, it will override `min_scale` and `max_scale`.
scales = [0.04, 0.08, 0.12, 0.24]
aspect_ratios = [0.5, 1.0, 2.0]  # The list of aspect ratios for the anchor boxes
two_boxes_for_ar1 = True  # Whether or not you want to generate two anchor boxes for aspect ratio 1
steps = None  # In case you'd like to set the step sizes for the anchor box grids manually; not recommended
offsets = None  # In case you'd like to set the offsets for the anchor box grids manually; not recommended
clip_boxes = False  # Whether or not to clip the anchor boxes to lie entirely within the image boundaries
variances = [1.0, 1.0, 1.0, 1.0]  # The list of variances by which the encoded target coordinates are scaled
normalize_coords = True  # Whether or not the model is supposed to use coordinates relative to the image size

# TODO: Set the path of the trained weights.
weights_path, resultfolder = './weightsniou0.3/hp2aconv6ssd512/hp2aconv6_ssd_epoch-500_loss-0.0358_val_loss-0.2121.h5', './detections/hp2aconv6ssd512/'

confidence_thresh = 0.5

# 1: Build the Keras model
K.clear_session()  # Clear previous models from memory.

model = hp2aconv6_ssd(image_size=(img_height, img_width, img_channels),
                      n_classes=n_classes,
                      mode='inference',
                      l2_regularization=0.0005,
                      scales=scales,
                      aspect_ratios_global=aspect_ratios,
                      aspect_ratios_per_layer=None,
                      two_boxes_for_ar1=two_boxes_for_ar1,
                      steps=steps,
                      offsets=offsets,
                      clip_boxes=clip_boxes,
                      variances=variances,
                      normalize_coords=normalize_coords,
                      subtract_mean=intensity_mean,
                      divide_by_stddev=intensity_range,
                      confidence_thresh=confidence_thresh,
                      iou_threshold=0.45,
                      top_k=200,
                      nms_max_output_size=400)


# 2: Load the trained weights into the model.

model.load_weights(weights_path, by_name=True)

# 3: Compile the model so that Keras won't complain the next time you load it.

adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)

model.compile(optimizer=adam, loss=ssd_loss.compute_loss)


dataset = DataGenerator(load_images_into_memory=False, hdf5_dataset_path=None)


# TODO: Set the paths to the dataset here.
test_labels_filename = dsfolder + dsname + '.txt'

classes = ['background',
           'open', 'short', 'mousebite', 'spur',
           'spurious_copper', 'pin_hole']

dataset.parse_pcb(labels_filename=test_labels_filename, img_channels=img_channels)

generator = dataset.generate_pcb(batch_size=batch_size,
                                 shuffle=False,
                                 argumentation='test',
                                 label_encoder=None,
                                 resize=[img_height, img_width],
                                 returns={'processed_images', 'image_ids', 'inverse_transform', 'original_images', 'original_labels'})

if not os.path.exists(resultfolder):
    os.makedirs(resultfolder)

n_images = dataset.get_dataset_size()
n_batches = int(ceil(n_images / batch_size))

results = [list() for _ in range(n_classes + 1)]
# Set the colors for the bounding boxes
colors = plt.cm.hsv(np.linspace(0, 1., n_classes + 1)).tolist()
colors = [(0, 0, 0), (230 / 255., 25 / 255., 75 / 255.), (60 / 255., 180 / 255., 75 / 255.),
          (255 / 255., 225 / 255., 25 / 255.), (0, 130 / 255., 200 / 255.),
          (245 / 255., 130 / 255., 48 / 255.), (145 / 255., 30 / 255., 180 / 255.), (70 / 255., 240 / 255., 240 / 255.)]
t1 = time.time()
for b in range(n_batches):
    batch_X, batch_image_ids, batch_inverse_transforms, batch_orig_images, batch_orig_labels = next(generator)
    # print(batch_X[0].shape, len(batch_orig_images))
    y_pred = model.predict(batch_X)

    # Perform confidence thresholding.
    y_pred_thresh = [y_pred[k][y_pred[k, :, 1] > confidence_thresh] for k in range(y_pred.shape[0])]

    # Convert the predicted box coordinates for the original images.
    y_pred_thresh_inv = apply_inverse_transforms(y_pred_thresh, batch_inverse_transforms)

    np.set_printoptions(precision=2, suppress=True, linewidth=90)

    # Iterate over all batch items.
    for k in range(len(y_pred_thresh_inv)):
        box = y_pred_thresh_inv[k]
        image_id = batch_image_ids[k]
        if isprint:
            print("Image:" + str(image_id))
            print("\nPredicted boxes:")
            print('   class   conf xmin   ymin   xmax   ymax')
            print(box)

            print("\nGround truth boxes:")
            print(batch_orig_labels[k])

        # Display the image and draw the predicted boxes onto it.

        if showimg or saveimg:
            plt.figure(figsize=(20, 12))
            plt.imshow(batch_orig_images[k][..., 0], cmap='gray')
            current_axis = plt.gca()

        for box in batch_orig_labels[k]:
            label, xmin, ymin, xmax, ymax = box
            if showimg or saveimg:
                color = colors[int(label)]
                label = '{}'.format(classes[int(label)])
                current_axis.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, color=color, fill=False, linewidth=2))
                current_axis.text(xmin, ymin - 6, label, size='x-large', color=color)

        if savetxt:
            resultfile = resultfolder + image_id + '.txt'
            f = open(resultfile, 'w')

        for box in y_pred_thresh_inv[k]:
            label, conf, xmin, ymin, xmax, ymax = box
            if savetxt:
                # f.write('%s %.8f %d %d %d %d\r' % (classes[int(label)], conf, xmin, ymin, xmax, ymax))
                # f.write('%d %.8f %d %d %d %d\r' % (label, xmin, ymin, xmax, ymax))
                f.write('%d %d %d %d %d %.8f\r' % (xmin, ymin, xmax, ymax, label, conf))
            if showimg or saveimg:
                color = colors[int(label)]
                label = '{}: {:.2f}'.format(classes[int(label)], conf)
                current_axis.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, color=color, fill=False, linewidth=1))
                current_axis.text(xmin, ymax + 16, label, size='x-large', color=color)
        if savetxt:
            f.close()

        if saveimg:
            resultfile = resultfolder + image_id + '.png'
            plt.axis('off')
            plt.savefig(resultfile, dpi=300, bbox_inches='tight')
        plt.show() if showimg else plt.close()

t2 = time.time()
t = t2 - t1
print("Detection time: %.4f, FPS: %d" % (t, int(n_images / t)))
