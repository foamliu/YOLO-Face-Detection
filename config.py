import numpy as np

image_h = image_w = image_size = 608
grid_h = grid_w = 19
num_channels = 3
grid_size = 32.0
num_box = 10
epsilon = 1e-6
jitter = True

score_threshold = 0.15  # real value, if [ highest class probability score < threshold], then get rid of the corresponding box
iou_threshold = 0.3  # real value, "intersection over union" threshold used for NMS filtering
anchors = [0.12, 0.20, 0.21, 0.38, 0.34, 0.63, 0.53, 0.99, 0.83, 1.47, 1.30, 2.23, 2.12, 3.16, 2.91, 5.28, 4.96, 7.00,
           7.85, 10.27]

labels = ['face']
num_classes = len(labels)
class_weights = np.ones(num_classes, dtype='float32')

train_image_folder = 'data/WIDER_train/images'
valid_image_folder = 'data/WIDER_val/images'
test_image_folder = 'data/WIDER_test/images'
train_annot_file = 'data/wider_face_split/wider_face_train_bbx_gt.txt'
valid_annot_file = 'data/wider_face_split/wider_face_val_bbx_gt.txt'
test_filelist_file = 'data/wider_face_split/wider_face_test_filelist.txt'

num_train_samples = 12880
num_valid_samples = 3226
verbose = 1
batch_size = 32
num_epochs = 1000
patience = 50

lambda_coord = 5.0
lambda_obj = 1.0
lambda_noobj = 0.5
lambda_class = 1.0

max_boxes = 10  # integer, maximum number of predicted boxes in an image
