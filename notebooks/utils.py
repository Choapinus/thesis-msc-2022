import cv2
import numpy as np

def load_image(path, colorspace='RGB'):
    color = colorspace.lower()
    spaces = {
        'rgb': cv2.COLOR_BGR2RGB,
        'hsv': cv2.COLOR_BGR2HSV,
        'hsv_full': cv2.COLOR_BGR2HSV_FULL,
        'gray': cv2.COLOR_BGR2GRAY,
        'lab': cv2.COLOR_BGR2LAB
    }

    if color not in spaces.keys(): 
        print(f'[WARNING] color space {colorspace} not supported')
        print(f'Supported list: {spaces.keys()}')
        print('Colorspace setted to RGB')
        color = 'rgb'
    
    image = cv2.imread(path)
    
    if image is None:
        return None
    else: return cv2.cvtColor(image, spaces[color])

def align(img, data):
    # In case the image has more than 1 face, find the biggest face
    biggest=0
    if data != []:
        for faces in data:
            box = faces['box']            
            # calculate the area in the image
            area = box[3]  * box[2]
            if area > biggest:
                biggest = area
                bbox = box                
                keypoints = faces['keypoints']
                left_eye = keypoints['left_eye']
                right_eye = keypoints['right_eye']                 
        
        lx,ly = left_eye        
        rx,ry = right_eye
        dx = rx-lx
        dy = ry-ly
        tan = dy/dx
        theta = np.arctan(tan)
        theta = np.degrees(theta)    
        img = rotate_bound(img, theta)        
        return img
    
    else:
        return None

def crop_image(img, data): 
    
    #y=box[1] h=box[3] x=box[0] w=box[2]   
    biggest=0
    
    if data != []:
        for faces in data:
            box = faces['box']            
            # calculate the area in the image
            area = box[3] * box[2]
            if area > biggest:
                biggest = area
                bbox = box 
        bbox[0] = 0 if bbox[0] < 0 else bbox[0]
        bbox[1] = 0 if bbox[1] < 0 else bbox[1]
        img = img[ bbox[1]: bbox[1] + bbox[3], bbox[0]: bbox[0] + bbox[2] ]
        return img
    else:
        return None

def rotate_bound(image, angle):
    # rotates an image by the degree angle
    # grab the dimensions of the image and then determine the center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    # grab the rotation matrix (applying the angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1]) 
    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin)) 
    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY 
    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH)) 