import struct
import numpy as np
import time

"""
Gets all labels from filename (assuming files are coming from http://yann.lecun.com/exdb/mnist/)
Returns a 1D array of labels that correspond in location with the training images
"""
def get_labels(filename):
    f = open(filename, 'rb')
    f.readline(4)
    num_labels = int.from_bytes(f.readline(4), byteorder='big')
    labels = np.zeros(num_labels)
    for i in range(num_labels):
        labels[i] = int.from_bytes(f.readline(1), byteorder='big')
    return labels

"""
Gets all images from filename (assuming files are coming from http://yann.lecun.com/exdb/mnist/)
Returns a 3D array of images with:
    1st index: image #
    2nd index: row # in image
    3rd index: col # in image
    The values stored are integers from 0 to 255 with 0 being black and 255 being white
"""
def get_images(filename):
    f = open(filename, 'rb')
    f.readline(4)
    num_images = int.from_bytes(f.readline(4), byteorder='big')
    num_rows = int.from_bytes(f.readline(4), byteorder='big')
    num_cols = int.from_bytes(f.readline(4), byteorder='big')
    images = np.zeros((num_images,num_rows,num_cols))
    for image in range(num_images):
        for row in range(num_rows):
            for col in range(num_cols):
                images[image][row][col] = int.from_bytes(f.readline(1), byteorder='big')
        # if(image%10000 == 0):
        #     print('Loaded %.1f%% of training images' % ((image/num_images)*100))
    f.close()
    return images

"""
Creates an image from the 3D images array (NOT THE 2D IMAGES ARRAY)
Index is the image # you want
Create file is image#.ppm
Use XnViewMP to view the image files after making them
"""
def make_image(index, images):
    filename = 'image' + str(index) + '.ppm'
    f = open(filename, 'w')
    header = 'P3\n28 28\n255\n'
    img = images[index]
    f.write(header)
    for i in range(28):
        for j in range(28):
            f.write((str(int(img[i][j]))+' ')*3)
    f.close()
    return

"""
Changes images array from 3D to 2D so that each row is one picture and each
column is a feature
"""
def convert_image_format(images):
    num_images = images.shape[0]
    rows = images.shape[1]
    cols = images.shape[2]
    new_data = np.zeros((num_images, rows*cols))
    for i in range(num_images):
        for row in range(rows):
            for col in range(cols):
                new_data[i][cols*row+col] = images[i][row][col]
    return new_data


"""
First, get all the data out of the files. Getting the training images takes the
longest, at about 1 minute.
"""
start = time.time()
print('-----------------------------------------------------------------------')
print('Starting to load training labels...')
training_labels = get_labels('train-labels-idx1-ubyte')
end = time.time()
print('Done! Took %.3f sec' % (end-start))

print('-----------------------------------------------------------------------')
start = time.time()
print('Starting to load training images...')
training_images = get_images('train-images-idx3-ubyte')
end = time.time()
print('Done! Took %.3f sec' % (end-start))

print('-----------------------------------------------------------------------')
start = time.time()
print('Starting to load test labels...')
test_labels = get_labels('t10k-labels-idx1-ubyte')
end = time.time()
print('Done! Took %.3f sec' % (end-start))

print('-----------------------------------------------------------------------')
start = time.time()
print('Starting to load test images...')
test_images = get_images('t10k-images-idx3-ubyte')
end = time.time()
print('Done! Took %.3f sec' % (end-start))
print('-----------------------------------------------------------------------')

"""
Then convert the image format from the 3D arrays, which are used for viewing the
images, to 2D arrays, which are used in the neural network.
"""
new_training_images = convert_image_format(training_images)
new_test_images = convert_image_format(test_images)

# make_image(0,imgs)
# make_image(1,imgs)
# make_image(2,imgs)
# make_image(3,imgs)
# make_image(4,imgs)
