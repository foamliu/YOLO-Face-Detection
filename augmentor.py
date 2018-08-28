import cv2 as cv
import numpy as np
from imgaug import augmenters as iaa

from config import image_h, image_w, num_classes, score_threshold, iou_threshold, anchors
from utils import decode_netout, draw_boxes

### augmentors by https://github.com/aleju/imgaug
sometimes = lambda aug: iaa.Sometimes(0.5, aug)

# Define our sequence of augmentation steps that will be applied to every image
# All augmenters with per_channel=0.5 will sample one value _per image_
# in 50% of all cases. In all other cases they will sample new values
# _per channel_.
aug_pipe = iaa.Sequential(
    [
        # apply the following augmenters to most images
        # iaa.Fliplr(0.5), # horizontally flip 50% of all images
        # iaa.Flipud(0.2), # vertically flip 20% of all images
        # sometimes(iaa.Crop(percent=(0, 0.1))), # crop images by 0-10% of their height/width
        sometimes(iaa.Affine(
            # scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}, # scale images to 80-120% of their size, individually per axis
            # translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}, # translate by -20 to +20 percent (per axis)
            # rotate=(-5, 5), # rotate by -45 to +45 degrees
            # shear=(-5, 5), # shear by -16 to +16 degrees
            # order=[0, 1], # use nearest neighbour or bilinear interpolation (fast)
            # cval=(0, 255), # if mode is constant, use a cval between 0 and 255
            # mode=ia.ALL # use any of scikit-image's warping modes (see 2nd image from the top for examples)
        )),
        # execute 0 to 5 of the following (less important) augmenters per image
        # don't execute all of them, as that would often be way too strong
        iaa.SomeOf((0, 5),
                   [
                       # sometimes(iaa.Superpixels(p_replace=(0, 1.0), n_segments=(20, 200))), # convert images into their superpixel representation
                       iaa.OneOf([
                           iaa.GaussianBlur((0, 0.5)),  # blur images with a sigma between 0 and 3.0
                           iaa.AverageBlur(k=(2, 3)),  # blur image using local means with kernel sizes between 2 and 3
                           iaa.MedianBlur(k=(3, 5)),
                           # blur image using local medians with kernel sizes between 2 and 7
                       ]),
                       iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)),  # sharpen images
                       # iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)), # emboss images
                       # search either for all edges or for directed edges
                       # sometimes(iaa.OneOf([
                       #    iaa.EdgeDetect(alpha=(0, 0.7)),
                       #    iaa.DirectedEdgeDetect(alpha=(0, 0.7), direction=(0.0, 1.0)),
                       # ])),
                       iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),
                       # add gaussian noise to images
                       iaa.OneOf([
                           iaa.Dropout((0.01, 0.1), per_channel=0.5),  # randomly remove up to 10% of the pixels
                           # iaa.CoarseDropout((0.03, 0.15), size_percent=(0.02, 0.05), per_channel=0.2),
                       ]),
                       # iaa.Invert(0.05, per_channel=True), # invert color channels
                       iaa.Add((-10, 10), per_channel=0.5),
                       # change brightness of images (by -10 to 10 of original value)
                       iaa.Multiply((0.5, 1.5), per_channel=0.5),
                       # change brightness of images (50-150% of original value)
                       iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5),  # improve or worsen the contrast
                       # iaa.Grayscale(alpha=(0.0, 1.0)),
                       # sometimes(iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)), # move pixels locally around (with random strengths)
                       # sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05))) # sometimes move parts of the image around
                   ],
                   random_order=True
                   )
    ],
    random_order=True
)


def aug_image(image, bboxes, jitter):
    orig_h, orig_w = image.shape[:2]

    if jitter:
        ### scale the image
        scale = np.random.uniform() / 10. + 1.
        image = cv.resize(image, (0, 0), fx=scale, fy=scale)

        ### translate the image
        max_offx = (scale - 1.) * orig_w
        max_offy = (scale - 1.) * orig_h
        offx = int(np.random.uniform() * max_offx)
        offy = int(np.random.uniform() * max_offy)

        image = image[offy: (offy + orig_h), offx: (offx + orig_w)]

        ### flip the image
        flip = np.random.binomial(1, .5)
        if flip > 0.5: image = cv.flip(image, 1)

        image = aug_pipe.augment_image(image)

        # resize the image to standard size
        image = cv.resize(image, (image_w, image_h))

        # fix object's position and size
        new_bboxes = []
        for bbox in bboxes:
            bx, by, bw, bh = bbox

            bx = int(bx * scale - offx)
            bw = int(bw * scale)

            bx = int(bx * float(image_w) / orig_w)
            bx = max(min(bx, image_w), 0)
            bw = int(bw * float(image_w) / orig_w)
            bw = max(min(bw, image_w), 0)

            by = int(by * scale - offy)
            bh = int(bh * scale)

            by = int(by * float(image_h) / orig_h)
            by = max(min(by, image_h), 0)
            bh = int(bh * float(image_h) / orig_h)
            bh = max(min(bh, image_h), 0)

            bx = bx / image_w
            bw = bw / image_w
            by = by / image_h
            bh = bh / image_h

            if flip > 0.5:
                bx = 1.0 - (bx + bw)

            new_bboxes.append((bx, by, bw, bh))
    else:
        # resize the image to standard size
        image = cv.resize(image, (image_w, image_h))
        new_bboxes = []
        for bbox in bboxes:
            bx, by, bw, bh = bbox
            bx = bx / orig_w
            bw = bw / orig_w
            by = by / orig_h
            bh = bh / orig_h
            new_bboxes.append((bx, by, bw, bh))

    return image, new_bboxes


def to_bboxes(bboxes):
    from utils import BoundBox
    new_bboxes = []
    for box in bboxes:
        x, y, w, h = box
        bbox = BoundBox(x, y, x + w, y + h)
        new_bboxes.append(bbox)
    return new_bboxes


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
