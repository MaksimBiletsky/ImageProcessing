import numpy as np
import cv2

kernels = {
    # 'box-3':    np.ones((3,3),np.float32) / 9,
    'box-10':   np.ones((10,10),np.float32) / 100,
    # 'box-15':   np.ones((15,15),np.float32) / 125
}

def own_filter2D(image, kernel):
    kern_x_shape, kern_y_shape = kernel.shape
    img_x_shape, img_y_shape = image.shape
    
    output_x = img_x_shape - kern_x_shape + 1
    output_y = img_y_shape - kern_y_shape + 1

    output = np.zeros((output_x, output_y))

    for y in range(img_y_shape):
        if y > img_y_shape - kern_y_shape:
            break
        for x in range(img_x_shape):
            if x > img_x_shape - kern_x_shape:
                break
            output[x, y] = (kernel * image[x: x + kern_x_shape, y: y + kern_y_shape]).sum()

    return output


def processImage(image):
    image = cv2.imread(image) 
    image = cv2.cvtColor(src=image, code=cv2.COLOR_BGR2GRAY)
    return image

image_name = 'detailed'
image = processImage(f'images/{image_name}.jpg')

for kernel_name, kernel in kernels.items():
    output = own_filter2D(image, kernel)
    cv2.imwrite(f'filtered_images/own_{kernel_name}_{image_name}.jpg', output)

    output = cv2.filter2D(image, -1, kernel)
    cv2.imwrite(f'filtered_images/opencv_{kernel_name}_{image_name}.jpg', output)