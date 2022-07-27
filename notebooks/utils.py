import cv2

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
    
    image = cv2.cvtColor(cv2.imread(path), spaces[color])
    return image