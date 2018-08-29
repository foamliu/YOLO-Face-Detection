import os

import cv2 as cv
import numpy as np
from keras.utils import Sequence

from augmentor import aug_image
from config import batch_size, image_h, image_w, grid_h, grid_w, num_classes, num_channels, num_box, grid_size, \
    train_image_folder, valid_image_folder, train_annot_file, valid_annot_file, anchors
from utils import BoundBox, bbox_iou, parse_annot

anchor_boxes = [BoundBox(0, 0, anchors[2 * i], anchors[2 * i + 1]) for i in range(int(len(anchors) // 2))]


def get_ground_truth(boxes):
    gt = np.zeros((grid_h, grid_w, num_box, 4 + 1 + num_classes), dtype=np.float32)

    for bbox in boxes:
        bx, by, bw, bh = bbox
        bx = 1.0 * bx * image_w
        by = 1.0 * by * image_h
        bw = 1.0 * bw * image_w
        bh = 1.0 * bh * image_h
        center_x = bx + bw / 2.
        center_x = center_x / grid_size
        center_y = by + bh / 2.
        center_y = center_y / grid_size
        cell_x = int(np.clip(np.floor(center_x), 0.0, (grid_w - 1)))
        cell_y = int(np.clip(np.floor(center_y), 0.0, (grid_h - 1)))
        center_w = bw / grid_size
        center_h = bh / grid_size
        box = [center_x, center_y, center_w, center_h]

        # find the anchor that best predicts this box
        best_anchor = -1
        max_iou = -1

        shifted_box = BoundBox(0,
                               0,
                               center_w,
                               center_h)

        for i in range(len(anchor_boxes)):
            anchor = anchor_boxes[i]
            iou = bbox_iou(shifted_box, anchor)

            if max_iou < iou:
                best_anchor = i
                max_iou = iou

        # assign ground truth x, y, w, h, confidence and class probs
        gt[cell_y, cell_x, best_anchor, 0] = 1.0
        gt[cell_y, cell_x, best_anchor, 1:5] = box
        gt[cell_y, cell_x, best_anchor, 5] = 1.0
    return gt


class DataGenSequence(Sequence):
    def __init__(self, usage):
        self.usage = usage

        if usage == 'train':
            self.image_folder = train_image_folder
            annot_file = train_annot_file
        else:
            self.image_folder = valid_image_folder
            annot_file = valid_annot_file

        self.annots = parse_annot(annot_file)
        self.num_samples = len(self.annots)

        np.random.shuffle(self.annots)

    def __len__(self):
        return int(np.ceil(self.num_samples / float(batch_size)))

    def __getitem__(self, idx):
        i = idx * batch_size

        length = min(batch_size, (self.num_samples - i))
        batch_x = np.empty((length, image_h, image_w, num_channels), dtype=np.float32)
        batch_y = np.empty((length, grid_h, grid_w, num_box, 4 + 1 + num_classes), dtype=np.float32)

        for i_batch in range(length):
            annot = self.annots[i + i_batch]
            filename = annot['filename']
            filename = os.path.join(self.image_folder, filename)
            image = cv.imread(filename)
            boxes = annot['bboxes']
            if self.usage == 'train':
                image, boxes = aug_image(image, boxes, jitter=True)
            else:
                image, boxes = aug_image(image, boxes, jitter=False)

            image = image[:, :, ::-1]
            batch_x[i_batch, :, :] = image / 255.
            batch_y[i_batch, :, :] = get_ground_truth(boxes)

        return batch_x, batch_y

    def on_epoch_end(self):
        np.random.shuffle(self.annots)


def train_gen():
    return DataGenSequence('train')


def valid_gen():
    return DataGenSequence('valid')


if __name__ == '__main__':
    from data_generator import DataGenSequence

    datagen = DataGenSequence('train')
    batch_inputs, batch_outputs = datagen.__getitem__(0)

    for i in range(10):
        image = batch_inputs[i]
        netout = batch_outputs[i]
        image = (image * 255.).astype(np.uint8)
        image = image[:, :, ::-1]
        cv.imwrite('images/imgaug_before_{}.png'.format(i), image)
        boxes = decode_netout(netout, anchors, num_classes, score_threshold, iou_threshold)
        print('boxes: ' + str(boxes))
        image = draw_boxes(image, boxes)
        cv.imwrite('images/imgaug_after_{}.png'.format(i), image)