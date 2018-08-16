# import the necessary packages
import os
import random

import cv2 as cv
import keras.backend as K
import numpy as np

from config import image_size, valid_image_folder, best_model, labels, anchors, num_classes
from model import build_model
from utils import ensure_folder, decode_netout, draw_boxes, get_best_model

if __name__ == '__main__':
    model = build_model()
    model.load_weights(get_best_model())

    test_path = valid_image_folder
    test_images = [f for f in os.listdir(test_path) if
                   os.path.isfile(os.path.join(test_path, f)) and f.endswith('.jpg')]
    num_samples = 20
    # random.seed(1)
    samples = random.sample(test_images, num_samples)

    ensure_folder('images')

    for i in range(len(samples)):
        image_name = samples[i]
        filename = os.path.join(test_path, image_name)
        print('Start processing image: {}'.format(filename))
        image_bgr = cv.imread(filename)
        image_shape = image_bgr.shape
        image_bgr = cv.resize(image_bgr, (image_size, image_size))
        image_rgb = image_bgr[:, :, ::-1]
        image_rgb = image_rgb / 255.
        image_input = np.expand_dims(image_rgb, 0).astype(np.float32)
        # [1, 13, 13, 5, 85]
        netout = model.predict(image_input)[0]
        boxes = decode_netout(netout, anchors, num_classes)
        image_bgr = draw_boxes(image_bgr, boxes, labels)
        cv.imwrite('images/{}_out.png'.format(i), image_bgr)

    K.clear_session()
