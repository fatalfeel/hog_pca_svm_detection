import numpy as np
import time
import random
import PIL
import cv2
import matplotlib.pyplot as plt
import glob
from skimage.feature import hog
from sklearn import svm
from sklearn import model_selection
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler

# Creating a SVC object
svc = svm.SVC()

# Find contours function of openCV will help us the find white regions.
def findContours_demo():
    example_mask = np.zeros((200, 200))
    example_mask[70:100, 60:120] = 255

    # contours, hierarchy = cv2.findContours(example_mask.astype(np.uint8), 1, 2)[-2:]
    cf_img, cf_cont, cf_hierarchy = cv2.findContours(example_mask.astype(np.uint8), 1, 2)
    for c in cf_cont:
        if cv2.contourArea(c) < 10 * 10:
            continue
        (x, y, w, h) = cv2.boundingRect(c)
        rgb_ver = cv2.cvtColor(example_mask.astype(np.uint8), cv2.COLOR_GRAY2RGB)
        im = cv2.rectangle(rgb_ver, (x, y), (x + w, y + h), (255, 0, 0), 3)

    fig = plt.figure(figsize=(12, 8))

    fig.add_subplot(1, 2, 1)
    plt.axis("off")
    plt.title('example block')
    plt.imshow(example_mask, cmap="gray")

    fig.add_subplot(1, 2, 2)
    plt.axis("off")
    plt.title('findContours')
    plt.imshow(im, cmap="gray")
    plt.show()

def slideExtract(image, windowSize=(96, 64), channel="RGB", step=12):
    # Converting to grayscale
    if channel == "RGB":    #PIL
        img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    elif channel == "BGR":  #opencv
        img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif channel.lower() != "grayscale" or channel.lower() != "gray":
        raise Exception("Invalid channel type")

    # We'll store coords and features in these lists
    coords = []
    features = []

    hIm, wIm = image.shape[:2]

    # W1 will start from 0 to end of image - window size
    # W2 will start from window size to end of image
    # We'll use step (stride) like convolution kernels.
    for w1, w2 in zip(range(0, wIm - windowSize[0], step), range(windowSize[0], wIm, step)):
        for h1, h2 in zip(range(0, hIm - windowSize[1], step), range(windowSize[1], hIm, step)):
            window = img[h1:h2, w1:w2]
            features_of_window = hog(window,
                                     orientations=9,
                                     pixels_per_cell=(16, 16),
                                     cells_per_block=(2, 2))

            coords.append((w1, w2, h1, h2))
            features.append(features_of_window)

    return (coords, np.asarray(features))

class Heatmap():
    def __init__(self, original_image):
        # Mask attribute is the heatmap initialized with zeros
        self.mask = np.zeros(original_image.shape[:2])

    # Increase value of region function will add some heat to heatmap
    def incValOfReg(self, coords):
        w1, w2, h1, h2 = coords
        self.mask[h1:h2, w1:w2] = self.mask[h1:h2, w1:w2] + 30

    # Decrease value of region function will remove some heat from heatmap
    # We'll use this function if a region considered negative
    def decValOfReg(self, coords):
        w1, w2, h1, h2 = coords
        self.mask[h1:h2, w1:w2] = self.mask[h1:h2, w1:w2] - 30

    def compileHeatmap(self):
        # As you know,pixel values must be between 0 and 255 (uint8)
        # Now we'll scale our values between 0 and 255 and convert it to uint8

        # Scaling between 0 and 1
        scaler = MinMaxScaler()

        self.mask = scaler.fit_transform(self.mask)

        # Scaling between 0 and 255
        self.mask = np.asarray(self.mask * 255).astype(np.uint8)

        # Now we'll threshold our mask, if a value is higher than 170, it will be white else
        # it will be black
        self.mask = cv2.inRange(self.mask, 170, 255)

        return self.mask

def detect(image):
    # Extracting features and initalizing heatmap
    coords, features = slideExtract(image)
    htmp = Heatmap(image)

    for i in range(len(features)):
        # If region is positive then add some heat
        decision = svc.predict([features[i]])

        if decision[0] == 1:
            htmp.incValOfReg(coords[i])
            # Else remove some heat
        else:
            htmp.decValOfReg(coords[i])

    # Compiling heatmap
    mask = htmp.compileHeatmap()

    #cont, _ = cv2.findContours(mask, 1, 2)[:2]
    fc_img, fc_cont, fc_hierarchy = cv2.findContours(mask, 1, 2)
    for c in fc_cont:
        # If a contour is small don't consider it
        if cv2.contourArea(c) < 70 * 70:
            continue

        (x, y, w, h) = cv2.boundingRect(c)
        image = cv2.rectangle(image, (x, y), (x + w, y + h), (255), 2)

    return image

if __name__ == '__main__':
    car_paths = glob.glob("./dataset/car_images" + "/*")[:5000]
    neg_paths = []

    for class_path in glob.glob("./dataset/natural_images" + "/*"):
        #if class_path != "./dataset/natural_images/car":
        paths       = random.choices(glob.glob(class_path + "/*"), k=700)
        neg_paths   = paths + neg_paths

    print("There are {} car images in the dataset".format(len(car_paths)))
    print("There are {} negative images in the dataset".format(len(neg_paths)))

    example_image = np.asarray(PIL.Image.open(car_paths[0]))
    hog_features, visualized = hog(example_image,
                                   orientations=9,
                                   pixels_per_cell=(16, 16),
                                   cells_per_block=(2, 2),
                                   visualize=True,
                                   channel_axis=2) #channel_axis=2 for black whilte display

    fig = plt.figure(figsize=(12, 6))

    fig.add_subplot(1, 2, 1)
    plt.axis("off")
    plt.title('image')
    plt.imshow(example_image)

    fig.add_subplot(1, 2, 2)
    plt.axis("off")
    plt.title('hog feature')
    plt.imshow(visualized, cmap="gray")
    plt.show()

    pos_images = []
    neg_images = []

    start = time.time()

    for car_path in car_paths:
        img = np.asarray(PIL.Image.open(car_path))
        # We don't have to use RGB channels to extract features, Grayscale is enough.
        img = cv2.cvtColor(cv2.resize(img, (96, 64)), cv2.COLOR_RGB2GRAY)
        img = hog(img,
                  orientations=9,
                  pixels_per_cell=(16, 16),
                  cells_per_block=(2, 2))

        pos_images.append(img)

    for neg_path in neg_paths:
        img = np.asarray(PIL.Image.open(neg_path))
        img = cv2.cvtColor(cv2.resize(img, (96, 64)), cv2.COLOR_RGB2GRAY)
        img = hog(img,
                  orientations=9,
                  pixels_per_cell=(16, 16),
                  cells_per_block=(2, 2))

        neg_images.append(img)

    pos_labels = np.ones(len(car_paths))
    neg_labels = np.zeros(len(neg_paths))
    x = np.asarray(pos_images + neg_images)
    y = np.asarray(list(pos_labels) + list(neg_labels))
    
    print("Shape of image set", x.shape)
    print("Shape of labels",    y.shape)

    processTime = round(time.time() - start, 2)
    print("Reading images and extracting features has taken {} seconds".format(processTime))

    x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.2, random_state=42)

    # We'll use Cross Validation Grid Search to find best parameters.
    # Classifier will be trained using each parameter
    svc.fit(x_train, y_train)

    y_pred = svc.predict(x_test)
    print("Accuracy score of model is ", accuracy_score(y_pred=y_pred, y_true=y_test) * 100)

    #test
    #example_image = np.asarray(PIL.Image.open("./dataset/car_images/Acura_ILX_2013_28_16_110_15_4_70_55_179_39_FWD_5_4_4dr_Mro.jpg"))
    #coords, features = slideExtract(example_image, channel="RGB")
    #findContours_demo()

    example_image   = np.asarray(PIL.Image.open("./dataset/car_images/Acura_MDX_2014_42_18_290_35_6_77_67_193_20_FWD_7_4_SUV_oiQ.jpg"))
    retimg          = detect(example_image)

    plt.title('result')
    plt.imshow(retimg, cmap=None)
    plt.show()

    input('press enter to continue')