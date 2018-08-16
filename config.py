import numpy as np

image_h = image_w = image_size = 416
grid_h = grid_w = 13
num_channels = 3
grid_size = 32
num_box = 2
epsilon = 1e-6

score_threshold = 0.5  # real value, if [ highest class probability score < threshold], then get rid of the corresponding box
iou_threshold = 0.4  # real value, "intersection over union" threshold used for NMS filtering
anchors = [0.18, 0.33, 1.04, 1.69]

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
