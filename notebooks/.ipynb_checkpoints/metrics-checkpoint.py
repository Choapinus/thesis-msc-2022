import tensorflow as tf

def GIOU(box1, box2):

    #  Receive coordinate information of prediction frame 
    box1_xy = box1[..., 0:2]  #  Receive the center point of all prediction frames xy
    box1_wh = box1[..., 2:4]  #  Receive the width and height of all prediction frames wh
    box1_wh_half = box1_wh // 2  #  Take half the width and height 
    box1_min = box1_xy - box1_wh_half  #  Coordinates of the upper left corner of the prediction box 
    box1_max = box1_xy + box1_wh_half  #  Coordinates of the lower right corner of the prediction box 
    #  The area of the prediction box w*h
    box1_area = box1_wh[..., 0] * box1_wh[..., 1]

    #  Receive the coordinate information of the real frame 
    box2_xy = box2[..., 0:2]  #  Receive the center point coordinates of all real boxes 
    box2_wh = box2[..., 2:4]  #  Receive the width and height of all real frames 
    box2_wh_half = box2_wh // 2  #  Take half the width and height 
    box2_min = box2_xy - box2_wh_half  #  The coordinates of the upper left corner of the real box 
    box2_max = box2_xy + box2_wh_half  #  The coordinates of the lower right corner of the real box 
    #  Real frame area w * h
    box2_area = box2_wh[..., 0] * box2_wh[..., 1]

    #  The intersection of two boxes 
    intersect_min = tf.maximum(box1_min, box2_min)  #  Coordinates of the upper left corner of the intersection 
    intersect_max = tf.minimum(box1_max, box2_max)  #  Coordinates of the lower right corner of the intersection 
    #  The width and height of the intersection 
    intersect_wh = intersect_max - intersect_min
    
    #  The area of the intersection iw*ih
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
    #  The area of the Union 
    union_area = box1_area + box2_area - intersect_area
    #  Calculation iou
    iou = intersect_area / (union_area + tf.keras.backend.epsilon())

    #  Calculate the smallest closed rectangular box that can contain prediction box and real box 
    enclose_min = tf.minimum(box1_min, box2_min)
    enclose_max = tf.maximum(box1_max, box2_max)
    #  The width and height of the smallest rectangle 
    enclose_wh = enclose_max - enclose_min
    #  The area of the closed-loop rectangle ew*eh
    enclose_area = enclose_wh[..., 0] * enclose_wh[..., 1]

    #  Calculation Giou
    giou = iou - (enclose_area - union_area) / (enclose_area + tf.keras.backend.epsilon())
    
    return iou, giou