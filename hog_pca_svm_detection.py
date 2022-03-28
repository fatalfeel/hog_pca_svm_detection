import glob
import random
import time
import PIL
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import hog
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

isPCA   = True
pca     = PCA(n_components=40)
svc     = svm.SVC()

def slideExtract(image, windowSize=(96, 64), channel="RGB", step=12):
    # Converting to grayscale
    if channel == "RGB":
        img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    elif channel == "BGR":
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
        if isPCA is True:
            x = np.array(pca.transform([features[i]]))
            decision = svc.predict(x)
        else:
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

#train pos use 96X160H96
if __name__ == '__main__':
    car_paths   = glob.glob("./dataset/car_images" + "/*")[:5000]
    neg_paths   = []
    pos_images  = []
    neg_images  = []

    for class_path in glob.glob("./dataset/natural_images" + "/*"):
        # if class_path != "./dataset/natural_images/car":
        paths = random.choices(glob.glob(class_path + "/*"), k=700)
        neg_paths = paths + neg_paths

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

    pos_labels  = np.ones(len(car_paths))
    neg_labels  = np.zeros(len(neg_paths))
    x           = np.asarray(pos_images + neg_images)
    y           = np.asarray(list(pos_labels) + list(neg_labels))

    start = time.time()

#########pca enable or disable this section########
    if isPCA is True:
        pca.fit(x)
        x = np.array(pca.fit_transform(x))

#########split features##########
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

#########svm##########
    clf     = svc.fit(x_train, y_train)
    print(clf.score(x_test, y_test))

########################test one picture time#####################
    example_image   = np.asarray(PIL.Image.open(car_paths[0]))
    example_image   = cv2.cvtColor(cv2.resize(example_image, (96, 64)), cv2.COLOR_RGB2GRAY)
    hog_image       = hog(example_image,
                          orientations=9,
                          pixels_per_cell=(16, 16),
                          cells_per_block=(2, 2))
    if isPCA is True:
        x = np.array(pca.transform([hog_image])) #[hog_image] to list array
        decision = svc.predict(x)
    else:
        decision = svc.predict([hog_image])

    processTime = round(time.time() - start, 2)
    print("Train and predict one picture time {} seconds".format(processTime))

########################test one car#####################
    example_image   = np.asarray(PIL.Image.open("./dataset/car_images/Acura_MDX_2014_42_18_290_35_6_77_67_193_20_FWD_7_4_SUV_oiQ.jpg"))
    retimg          = detect(example_image)
    plt.title('result')
    plt.imshow(retimg, cmap=None)
    plt.show()

    processTime = round(time.time() - start, 2)
    print("Train and detection car time {} seconds".format(processTime))

    input('press enter to continue')