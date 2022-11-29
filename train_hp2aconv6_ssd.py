import os
from math import ceil
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, CSVLogger, LearningRateScheduler
from keras import backend as K

from models.keras_hp2aconv6_ssd import hp2aconv6_ssd
from keras_loss_function.keras_ssd_loss import SSDLoss

from ssd_encoder_decoder.ssd_input_encoder import SSDInputEncoder
from data_generator.object_detection_2d_data_generator import DataGenerator

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # CPU
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # GPU

niou = 0.3
dsfolder, traindsname, validdsname = './datasets/PCBData/', 'trainval', 'test'
img_height = 512  # Height of the model input images
img_width = 512  # Width of the model input images
img_channels = 2  # Number of color channels of the input images
intensity_mean = None  # Set this to your preference (maybe `None`). The current settings transform the input pixel values to the interval `[-1,1]`.
intensity_range = None  # Set this to your preference (maybe `None`). The current settings transform the input pixel values to the interval `[-1,1]`.
n_classes = 6  # Number of positive classes
scales = [0.04, 0.08, 0.12, 0.24]  # An explicit list of anchor box scaling factors. If this is passed, it will override `min_scale` and `max_scale`.
aspect_ratios = [0.5, 1.0, 2.0]  # The list of aspect ratios for the anchor boxes
two_boxes_for_ar1 = True  # Whether or not you want to generate two anchor boxes for aspect ratio 1
steps = None  # In case you'd like to set the step sizes for the anchor box grids manually; not recommended
offsets = None  # In case you'd like to set the offsets for the anchor box grids manually; not recommended
clip_boxes = False  # Whether or not to clip the anchor boxes to lie entirely within the image boundaries
variances = [1.0, 1.0, 1.0, 1.0]  # The list of variances by which the encoded target coordinates are scaled
normalize_coords = True  # Whether or not the model is supposed to use coordinates relative to the image size

# 1: Build the Keras model

K.clear_session()  # Clear previous models from memory.

model = hp2aconv6_ssd(image_size=(img_height, img_width, img_channels),
                      n_classes=n_classes,
                      mode='training',
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
                      divide_by_stddev=intensity_range)

# 2: Optional: Load some weights
logfolder = './weightsniou%.1f/hp2aconv6ssd%d/' % (niou, img_height)
logfolder = './weightsniou%.1f/hp2aconv6ssd%dnostand/' % (niou, img_height)

# model.load_weights('./ssd7_weights.h5', by_name=True)

# 3: Instantiate an Adam optimizer and the SSD loss function and compile the model

adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)

model.compile(optimizer=adam, loss=ssd_loss.compute_loss)

# 1: Instantiate two `DataGenerator` objects: One for training, one for validation.

# Optional: If you have enough memory, consider loading the images into memory for the reasons explained above.

train_dataset = DataGenerator(load_images_into_memory=True, hdf5_dataset_path=None)
valid_dataset = DataGenerator(load_images_into_memory=True, hdf5_dataset_path=None)

# 2: Parse the image and label lists for the training and validation datasets.
train_labels_filename = dsfolder + traindsname + '.txt'
valid_labels_filename = dsfolder + validdsname + '.txt'

train_dataset.parse_pcb(labels_filename=train_labels_filename, img_channels=img_channels)
valid_dataset.parse_pcb(labels_filename=valid_labels_filename, img_channels=img_channels)


def lr_scheduler(epoch):
    lr_base = 1e-3
    step = 100
    decay_rate = 0.33
    delta = epoch // step
    return lr_base * (decay_rate**delta)
    # return lr_base

# Optional: Convert the dataset into an HDF5 dataset. This will require more disk space, but will
# speed up the training. Doing this is not relevant in case you activated the `load_images_into_memory`
# option in the constructor, because in that cas the images are in memory already anyway. If you don't
# want to create HDF5 datasets, comment out the subsequent two function calls.


# 3: Set the batch size.

batch_size = 10

# 5: Instantiate an encoder that can encode ground truth labels into the format needed by the SSD loss function.

# The encoder constructor needs the spatial dimensions of the model's predictor layers to create the anchor boxes.
predictor_sizes = [model.get_layer('classes4').output_shape[1:3],
                   model.get_layer('classes5').output_shape[1:3],
                   model.get_layer('classes6').output_shape[1:3]]

ssd_input_encoder = SSDInputEncoder(img_height=img_height,
                                    img_width=img_width,
                                    n_classes=n_classes,
                                    predictor_sizes=predictor_sizes,
                                    scales=scales,
                                    aspect_ratios_global=aspect_ratios,
                                    two_boxes_for_ar1=two_boxes_for_ar1,
                                    steps=steps,
                                    offsets=offsets,
                                    clip_boxes=clip_boxes,
                                    variances=variances,
                                    matching_type='multi',
                                    pos_iou_threshold=0.5,
                                    neg_iou_limit=niou,
                                    normalize_coords=normalize_coords)

# 6: Create the generator handles that will be passed to Keras' `fit_generator()` function.

train_generator = train_dataset.generate_pcb(batch_size=batch_size,
                                             shuffle=True,
                                             argumentation='train',
                                             label_encoder=ssd_input_encoder,
                                             resize=[img_height, img_width])
                                             # random_crop=img_height)

valid_generator = valid_dataset.generate_pcb(batch_size=batch_size,
                                             shuffle=False,
                                             argumentation='valid',
                                             label_encoder=ssd_input_encoder,
                                             resize=[img_height, img_width])

# Get the number of samples in the training and validations datasets.
train_dataset_size = train_dataset.get_dataset_size()
valid_dataset_size = valid_dataset.get_dataset_size()

print("Number of images in the training dataset:\t{:>6}".format(train_dataset_size))
print("Number of images in the validation dataset:\t{:>6}".format(valid_dataset_size))


# TODO: Set the filepath under which you want to save the weights.
model_checkpoint = ModelCheckpoint(filepath=logfolder + 'hp2aconv6_ssd_epoch-{epoch:02d}_loss-{loss:.4f}_val_loss-{val_loss:.4f}.h5',
                                   monitor='val_loss',
                                   verbose=1,
                                   save_best_only=False,
                                   save_weights_only=True,
                                   mode='auto',
                                   period=1)

csv_logger = CSVLogger(filename=logfolder + 'hp2aconv6_ssd' + str(img_height) + '_training_log.csv',
                       separator=',',
                       append=True)

learning_rate = LearningRateScheduler(lr_scheduler)

callbacks = [model_checkpoint, csv_logger, learning_rate]

# If you're resuming a previous training, set `initial_epoch` and `final_epoch` accordingly.
initial_epoch = 0
final_epoch = 500
steps_per_epoch = int(train_dataset_size / batch_size)


history = model.fit_generator(generator=train_generator,
                              steps_per_epoch=steps_per_epoch,
                              epochs=final_epoch,
                              verbose=2,
                              callbacks=callbacks,
                              validation_data=valid_generator,
                              validation_steps=ceil(valid_dataset_size / batch_size),
                              initial_epoch=initial_epoch)
