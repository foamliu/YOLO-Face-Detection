import keras.backend as K
import numpy as np
import tensorflow as tf

from config import image_size, grid_h, grid_w, grid_size, class_weights, anchors, batch_size, num_box, epsilon
from config import lambda_coord, lambda_noobj, lambda_class, lambda_obj


def yolo_loss(y_true, y_pred):
    # [None, 13, 13, 5]
    mask_shape = tf.shape(y_true)[:4]
    length = mask_shape[0]

    cell_x = tf.to_float(tf.reshape(tf.tile(tf.range(grid_w), [grid_h]), (1, grid_h, grid_w, 1, 1)))
    cell_y = tf.transpose(cell_x, (0, 2, 1, 3, 4))

    cell_grid = tf.tile(tf.concat([cell_x, cell_y], -1), [length, 1, 1, 5, 1])

    # [None, 13, 13, 5]
    coord_mask = tf.zeros(mask_shape)
    conf_mask = tf.zeros(mask_shape)
    class_mask = tf.zeros(mask_shape)

    seen = tf.Variable(0.)
    total_recall = tf.Variable(0.)

    """
    Adjust ground truth
    """
    # adjust confidence
    # [None, 13, 13, 5]
    box_conf = y_true[..., 0]
    # adjust x and y
    # [None, 13, 13, 5, 2]
    box_xy = y_true[..., 1:3]   # relative position to the containing cell
    # adjust w and h
    # [None, 13, 13, 5, 2]
    box_wh = y_true[..., 3:5]   # number of cells across, horizontally and vertically
    # adjust class probabilities
    # [None, 13, 13, 5]
    box_class = tf.argmax(y_true[..., 5:], -1)

    box_wh_half = box_wh / 2.
    box_mins = box_xy - box_wh_half
    box_maxes = box_xy + box_wh_half

    """
        Adjust prediction
    """
    # adjust confidence
    # [None, 13, 13, 5]
    box_conf_hat = tf.sigmoid(y_pred[..., 0])
    # adjust x and y
    # [None, 13, 13, 5, 2]
    box_xy_hat = tf.sigmoid(y_pred[..., 1:3]) + cell_grid
    # adjust w and h
    # [None, 13, 13, 5, 2]
    box_wh_hat = tf.exp(y_pred[..., 3:5]) * np.reshape(anchors, [1, 1, 1, num_box, 2])
    # adjust class probabilities
    # [None, 13, 13, 5, 80]
    box_class_hat = y_pred[..., 5:]

    box_wh_half_hat = box_wh_hat / 2.
    box_mins_hat = box_xy_hat - box_wh_half_hat
    box_maxes_hat = box_xy_hat + box_wh_half_hat


    # [None, 13, 13, 5, 2]
    intersect_mins = tf.maximum(box_mins_hat, box_mins)
    intersect_maxes = tf.minimum(box_maxes_hat, box_maxes)
    intersect_wh = tf.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]

    # [None, 13, 13, 5]
    true_areas = box_wh[..., 0] * box_wh[..., 1]
    pred_areas = box_wh_hat[..., 0] * box_wh_hat[..., 1]

    # [None, 13, 13, 5]
    union_areas = pred_areas + true_areas - intersect_areas
    iou_scores = tf.truediv(intersect_areas, union_areas + 1e-6)

    # [None, 13, 13, 5]
    box_conf = iou_scores * box_conf


    """
        Determine the masks
    """
    # the position of the ground truth boxes (the predictors)
    # [None, 13, 13, 5, 1]
    coord_mask = K.expand_dims(y_true[..., 0], axis=-1) * lambda_coord
    # [None, 13, 13, 5]
    best_ious = iou_scores

    """
        confidence mask: penelize predictors + penalize boxes with low IOU
        penalize the confidence of the boxes, which have IOU with some ground truth box < 0.6
    """
    # [None, 13, 13, 5]
    conf_mask = conf_mask + tf.to_float(best_ious < 0.6) * (1 - y_true[..., 0]) * lambda_noobj
    # penalize the confidence of the boxes, which are responsible for corresponding ground truth box
    conf_mask = conf_mask + y_true[..., 0] * lambda_obj
    # [None, 13, 13, 5]
    class_mask = y_true[..., 0] * tf.gather(class_weights, box_class) * lambda_class

    # [None, 13, 13, 5] -> integer
    nb_coord_box = tf.reduce_sum(tf.to_float(coord_mask > 0.0))
    nb_conf_box = tf.reduce_sum(tf.to_float(conf_mask > 0.0))
    nb_class_box = tf.reduce_sum(tf.to_float(class_mask > 0.0))

    # [None, 13, 13, 5, 2]
    loss_xy = tf.reduce_sum(tf.square(box_xy - box_xy_hat) * coord_mask) / (nb_coord_box + epsilon) / 2.
    # [None, 13, 13, 5, 2]
    loss_wh = tf.reduce_sum(tf.square(box_wh - box_wh_hat) * coord_mask) / (nb_coord_box + epsilon) / 2.
    # [None, 13, 13, 5]
    loss_conf = tf.reduce_sum(tf.square(box_conf - box_conf_hat) * conf_mask) / (nb_conf_box + epsilon) / 2.
    # [None, 13, 13, 5]
    loss_class = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=box_class, logits=box_class_hat)
    loss_class = tf.reduce_sum(loss_class * class_mask) / (nb_class_box + epsilon)

    loss = loss_xy + loss_wh + loss_conf + loss_class

    nb_true_box = tf.reduce_sum(y_true[..., 0])
    nb_pred_box = tf.reduce_sum(tf.to_float(box_conf > 0.5) * tf.to_float(box_conf_hat > 0.3))

    """
    Debugging code
    """
    current_recall = nb_pred_box / (nb_true_box + 1e-6)
    # total_recall = tf.assign_add(total_recall, current_recall)

    loss = tf.Print(loss, [tf.zeros((1))], first_n=10, message='Dummy Line \t', summarize=1000)
    loss = tf.Print(loss, [loss_xy], first_n=10, message='Loss XY \t', summarize=1000)
    loss = tf.Print(loss, [loss_wh], first_n=10, message='Loss WH \t', summarize=1000)
    loss = tf.Print(loss, [loss_conf], first_n=10, message='Loss Conf \t', summarize=1000)
    loss = tf.Print(loss, [loss_class], first_n=10, message='Loss Class \t', summarize=1000)
    loss = tf.Print(loss, [loss], first_n=10, message='Total Loss \t', summarize=1000)
    loss = tf.Print(loss, [current_recall], first_n=10, message='Current Recall \t', summarize=1000)
    # loss = tf.Print(loss, [total_recall / seen], first_n=10, message='Average Recall \t', summarize=1000)

    return loss


def yolo_filter_boxes(box_confidence, boxes, box_class_probs, threshold=.6):
    """Filters YOLO boxes by thresholding on object and class confidence.

    Arguments:
    box_confidence -- tensor of shape (13, 13, 1)
    boxes -- tensor of shape (13, 13, 4)
    box_class_probs -- tensor of shape (13, 13, 80)
    threshold -- real value, if [ highest class probability score < threshold], then get rid of the corresponding box

    Returns:
    scores -- tensor of shape (None,), containing the class probability score for selected boxes
    boxes -- tensor of shape (None, 4), containing (b_x, b_y, b_h, b_w) coordinates of selected boxes
    classes -- tensor of shape (None,), containing the index of the class detected by the selected boxes

    Note: "None" is here because you don't know the exact number of selected boxes, as it depends on the threshold.
    For example, the actual output size of scores would be (10,) if there are 10 boxes.
    """

    # Step 1: Compute box scores
    box_scores = box_confidence * box_class_probs  # [13, 13, 80]
    print('box_scores.shape: ' + str(box_scores.shape))

    # Step 2: Find the box_classes thanks to the max box_scores, keep track of the corresponding score
    box_classes = np.argmax(box_scores, axis=-1)  # [13, 13]
    box_classes = np.expand_dims(box_classes, axis=-1)  # [13, 13, 1]
    print('box_classes.shape: ' + str(box_classes.shape))
    box_class_scores = np.max(box_scores, axis=-1, keepdims=True)  # [13, 13, 1]
    print('box_class_scores.shape: ' + str(box_class_scores.shape))
    print('np.mean(box_class_scores): ' + str(np.mean(box_class_scores)))
    print('np.max(box_class_scores): ' + str(np.max(box_class_scores)))
    print('np.std(box_class_scores): ' + str(np.std(box_class_scores)))

    # Step 3: Create a filtering mask based on "box_class_scores" by using "threshold". The mask should have the
    # same dimension as box_class_scores, and be True for the boxes you want to keep (with probability >= threshold)
    filtering_mask = box_class_scores >= threshold  # [13, 13, 1]
    # print('filtering_mask: ' + str(filtering_mask))
    print('filtering_mask.shape: ' + str(filtering_mask.shape))

    # Step 4: Apply the mask to scores, boxes and classes
    scores = box_class_scores[filtering_mask]
    print('scores.shape: ' + str(scores.shape))  # [num_remain]
    boxes = boxes[np.repeat(filtering_mask, 4, axis=2)]  # [num_remain x 4]
    print('boxes.shape: ' + str(boxes.shape))
    classes = box_classes[filtering_mask]  # [num_remain]
    print('classes.shape: ' + str(classes.shape))

    return scores, boxes, classes


# box_xy: [14, 14, 2]
# box_wh: [14, 14, 2]
def yolo_boxes_to_corners(box_xy, box_wh):
    """Convert YOLO box predictions to bounding box corners."""
    box_mins = box_xy - (box_wh / 2.)
    box_maxes = box_xy + (box_wh / 2.)
    box_mins = np.clip(box_mins, 0, image_size - 1)
    box_maxes = np.clip(box_maxes, 0, image_size - 1)

    # [14, 14, 4]
    result = np.concatenate([
        box_mins[..., 0:1],  # x_min
        box_mins[..., 1:2],  # y_min
        box_maxes[..., 0:1],  # x_max
        box_maxes[..., 1:2]  # y_max
    ], axis=-1)
    print('result.shape: ' + str(result.shape))
    return result


def yolo_scale_box_xy(box_xy):
    result = np.zeros_like(box_xy)
    # shape = 14, 14, 2
    for cell_y in range(grid_h):
        for cell_x in range(grid_w):
            bx = box_xy[cell_y, cell_x, 0]
            by = box_xy[cell_y, cell_x, 1]
            temp_x = (cell_x + bx) * grid_size
            temp_y = (cell_y + by) * grid_size
            result[cell_y, cell_x, 0] = temp_x
            result[cell_y, cell_x, 1] = temp_y
    return result


def yolo_non_max_suppression(scores, boxes, classes, max_boxes=10, iou_threshold=0.5):
    """
    Applies Non-max suppression (NMS) to set of boxes

    Arguments:
    scores -- tensor of shape (None,), output of yolo_filter_boxes()
    boxes -- tensor of shape (None, 4), output of yolo_filter_boxes() that have been scaled to the image size (see later)
    classes -- tensor of shape (None,), output of yolo_filter_boxes()
    max_boxes -- integer, maximum number of predicted boxes you'd like
    iou_threshold -- real value, "intersection over union" threshold used for NMS filtering

    Returns:
    scores -- tensor of shape (, None), predicted score for each box
    boxes -- tensor of shape (4, None), predicted box coordinates
    classes -- tensor of shape (, None), predicted class for each box

    Note: The "None" dimension of the output tensors has obviously to be less than max_boxes. Note also that this
    function will transpose the shapes of scores, boxes, classes. This is made for convenience.
    """

    max_boxes_tensor = K.variable(max_boxes, dtype='int32')  # tensor to be used in tf.image.non_max_suppression()
    K.get_session().run(tf.variables_initializer([max_boxes_tensor]))  # initialize variable max_boxes_tensor

    # Use tf.image.non_max_suppression() to get the list of indices corresponding to boxes you keep
    nms_indices = tf.image.non_max_suppression(boxes, scores, max_boxes, iou_threshold)

    # Use K.gather() to select only nms_indices from scores, boxes and classes
    scores = K.gather(scores, nms_indices)
    boxes = K.gather(boxes, nms_indices)
    classes = K.gather(classes, nms_indices)

    return K.eval(scores), K.eval(boxes), K.eval(classes)
