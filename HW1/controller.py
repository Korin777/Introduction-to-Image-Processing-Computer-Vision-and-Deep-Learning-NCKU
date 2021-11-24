
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QFileDialog
import cv2 as cv
import numpy as np
from numpy.core.fromnumeric import size
from numpy.lib.function_base import average
from tensorflow.python.ops.gen_batch_ops import batch
from tensorflow.python.util.nest import flatten
from UI import Ui_MainWindow
from scipy import signal
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import MaxPool2D, MaxPooling2D, Conv2D, Flatten, Dense, Dropout
from tensorflow import keras

class MainWindow_controller(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.setup_control()
        self.value = 0
        self.img = 0
        self.h = 0
        self.w = 0
        self.center = (0,0)
        self.label = np.array(["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"])
        self.count = 0
        import os
        from keras import backend
        from keras.datasets.cifar import load_batch
        path = "./cifar-10-batches-py/"
        num_train_samples = 50000
        self.x_train = np.empty((num_train_samples, 3, 32, 32), dtype='uint8')
        self.y_train = np.empty((num_train_samples,), dtype='uint8')

        for i in range(1, 6):
            fpath = os.path.join(path, 'data_batch_' + str(i))
            (self.x_train[(i - 1) * 10000:i * 10000, :, :, :],
            self.y_train[(i - 1) * 10000:i * 10000]) = load_batch(fpath)

        fpath = os.path.join(path, 'test_batch')
        self.x_test, self.y_test = load_batch(fpath)

        self.y_train = np.reshape(self.y_train, (len(self.y_train), 1))
        self.y_test = np.reshape(self.y_test, (len(self.y_test), 1))

        if backend.image_data_format() == 'channels_last':
            self.x_train = self.x_train.transpose(0, 2, 3, 1)
            self.x_test = self.x_test.transpose(0, 2, 3, 1)

        self.x_test = self.x_test.astype(self.x_train.dtype)
        self.y_test = self.y_test.astype(self.y_train.dtype)

        self.batch_size = 128
        self.learning_rate = 0.1
        self.optimizer = 'SGD'
        self.model = tf.keras.models.load_model('./model/MyVGG16.h5')
        

    def setup_control(self):

        # 1. Image Processing
        self.ui.pushButton.clicked.connect(self.LoadImage)
        self.ui.pushButton_2.clicked.connect(self.ColorSeperation)
        self.ui.pushButton_3.clicked.connect(self.ColorTransformation)
        self.ui.pushButton_4.clicked.connect(self.Blending)
        # 2. Image Smoothing
        self.ui.pushButton_5.clicked.connect(self.GaussianBlur)
        self.ui.pushButton_6.clicked.connect(self.BilateralFilter)
        self.ui.pushButton_7.clicked.connect(self.MedianFilter)
        # 3. Edge Detection
        self.ui.pushButton_8.clicked.connect(self.MyGaussianBlur)
        self.ui.pushButton_9.clicked.connect(self.MySobelX)
        self.ui.pushButton_10.clicked.connect(self.MySobelY)
        self.ui.pushButton_11.clicked.connect(self.MyMagnitude)
        # 4. Transforms
        self.ui.pushButton_12.clicked.connect(self.Resize)
        self.ui.pushButton_13.clicked.connect(self.Translation)
        self.ui.pushButton_14.clicked.connect(self.RotationScaling)
        self.ui.pushButton_15.clicked.connect(self.Shearing)
        # 5. Training Cifar10 Classifier Using VGG16
        self.ui.pushButton_16.clicked.connect(self.ShowTrainImage)
        self.ui.pushButton_17.clicked.connect(self.ShowHyperparameter)
        self.ui.pushButton_18.clicked.connect(self.ShowModelStructure)
        self.ui.pushButton_19.clicked.connect(self.Train)
        self.ui.pushButton_20.clicked.connect(self.Predict)



    # 1. Image Processing
    def LoadImage(self):
        filename = self.open_file()
        if filename == '':
            return
        img = cv.imread(filename)
        cv.imshow(filename,img)
        print("Height :",img.shape[0])
        print("Width :",img.shape[1])
    def ColorSeperation(self):
        filename = self.open_file()
        if filename == '':
            return
        img = cv.imread(filename)
        b, g, r = cv.split(img)
        blank = np.zeros(img.shape[:2],dtype='uint8')
        cv.imshow("B",cv.merge([b,blank,blank]))
        cv.imshow("G",cv.merge([blank,g,blank]))
        cv.imshow("R",cv.merge([blank,blank,r]))
    def ColorTransformation(self):
        filename = self.open_file()
        if filename == '':
            return
        img = cv.imread(filename)
        b, g, r = cv.split(img)
        aw = np.zeros(img.shape[:2],dtype='uint8')
        aw =  r/3 + g/3 + b/3
        aw = np.around(aw)
        aw = aw.astype(np.uint8)
        cv.imshow("gray sacle",cv.cvtColor(img,cv.COLOR_BGR2GRAY))
        cv.imshow("average weighted",aw)
    def Blending(self):
        filename = self.open_file()
        filename2 = self.open_file()
        if filename == '' or filename2 == '':
            return
        img = cv.imread(filename)
        img2 = cv.imread(filename2)
        cv.namedWindow('Blending')
        cv.createTrackbar("Blend","Blending",0,255,self.update)
        while 1:
            img3 = cv.addWeighted(img,self.value/256,img2,(256-self.value)/256,0)
            cv.imshow("Blending",img3)
            k = cv.waitKey(1) & 0xFF
            if k == 27: # ESC
                break
    # 2. Image Smoothing
    def GaussianBlur(self):
        filename = self.open_file()
        if filename == '':
            return
        img = cv.imread(filename)
        gb = cv.GaussianBlur(img,(5,5),0)
        cv.imshow("Gaussian Blur",gb)
    def BilateralFilter(self):
        filename = self.open_file()
        if filename == '':
            return
        img = cv.imread(filename)
        bf = cv.bilateralFilter(img,9,90,90)
        cv.imshow("Bilateral Filter",bf)
    def MedianFilter(self):
        filename = self.open_file()
        if filename == '':
            return
        img = cv.imread(filename)
        mf = cv.medianBlur(img,3)
        mf2 = cv.medianBlur(img,5)
        cv.imshow("Median Filter(3*3)",mf)
        cv.imshow("Median Filter(5*5)",mf2)
    # 3. Edge Detection
    def MyGaussianBlur(self):
        x,y = np.mgrid[-1:2,-1:2]
        g = np.exp(-(x**2 + y**2))
        g = g / g.sum()
        filename = self.open_file()
        if filename == '':
            return
        img = cv.imread(filename)
        cv.imshow(filename,img)
        img = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
        cv.imshow("Grayscale",img)
        img = signal.convolve2d(img,g,boundary='symm',mode='same')
        img = np.round(img)
        img = img.astype(np.uint8)
        cv.imshow("Gaussian Blur",img)
    def MySobelX(self):
        x,y = np.mgrid[-1:2,-1:2]
        g = np.exp(-(x**2 + y**2))
        g = g / g.sum()
        s = np.array([(-1,0,1),
                     (-2,0,2),     
                     (-1,0,1),])
        filename = self.open_file()
        if filename == '':
            return
        img = signal.convolve2d(cv.cvtColor(cv.imread(filename),cv.COLOR_BGR2GRAY),g,boundary='symm',mode='same')
        img = np.round(img)
        img = img.astype(np.uint8)
        cv.imshow("Gaussian Blur",img)
        img = signal.convolve2d(img,s,boundary='symm',mode='same')
        img = np.uint8(np.absolute(np.round(img)))
        cv.imshow("Sobel X",img)
    def MySobelY(self):
        x,y = np.mgrid[-1:2,-1:2]
        g = np.exp(-(x**2 + y**2))
        g = g / g.sum()
        s = np.array([(1,2,1),
                      (0,0,0),     
                      (-1,-2,-1),])
        filename = self.open_file()
        if filename == '':
            return
        img = signal.convolve2d(cv.cvtColor(cv.imread(filename),cv.COLOR_BGR2GRAY),g,boundary='symm',mode='same')
        img = np.round(img)
        img = img.astype(np.uint8)
        cv.imshow("Gaussian Blur",img)
        img = signal.convolve2d(img,s,boundary='symm',mode='same')
        img = np.uint8(np.absolute(np.round(img)))
        cv.imshow("Sobel Y",img)
    def MyMagnitude(self):
        filename = self.open_file()
        if filename == '':
            return
        img = cv.imread(filename)
        x,y = np.mgrid[-1:2,-1:2]
        g = np.exp(-(x**2 + y**2))
        g = g / g.sum()
        sx = np.array([(-1,0,1),
                (-2,0,2),     
                (-1,0,1),])
        sy = np.array([(1,2,1),
                (0,0,0),     
                (-1,-2,-1),])
        img = signal.convolve2d(cv.cvtColor(cv.imread(filename),cv.COLOR_BGR2GRAY),g,boundary='symm',mode='same')  
        img = np.uint8(np.round(img))
        imgsx = signal.convolve2d(img,sx,boundary='symm',mode='same')
        imgsx = np.uint8(np.absolute(np.round(imgsx)))
        imgsy = signal.convolve2d(img,sy,boundary='symm',mode='same')
        imgsy = np.uint8(np.absolute(np.round(imgsy)))
        imgmag = np.uint8(np.round(np.sqrt(np.uint16(imgsx)**2 + np.uint16(imgsy)**2)))
        cv.imshow("Sobel x",imgsx)
        cv.imshow("Sobel Y",imgsy)
        cv.imshow("Magnitude",imgmag)
    # 4. Transforms
    def Resize(self):
        filename = self.open_file()
        if filename == '':
            return
        self.img = cv.imread(filename)
        self.w = int(self.ui.lineEdit.text())
        self.h = int(self.ui.lineEdit_2.text())
        self.center = (int(self.w/2),int(self.h/2))
        self.img = cv.resize(self.img,(self.w,self.h))
        cv.imshow("Resize",self.img)
    def Translation(self):
        x = int(self.ui.lineEdit.text())
        y = int(self.ui.lineEdit_2.text())
        self.center = (self.center[0] + x,self.center[1] + y)
        M = np.float32([[1,0,x],
                        [0,1,y]])
        self.img = cv.warpAffine(self.img,M,(int(self.ui.lineEdit_7.text()),int(self.ui.lineEdit_8.text())))
        cv.imshow("Translation",self.img)
    def RotationScaling(self):
        angle = float(self.ui.lineEdit.text())
        scale = float(self.ui.lineEdit_2.text())
        trans = cv.getRotationMatrix2D(self.center,angle,scale)
        self.img = cv.warpAffine(self.img,trans,(int(self.ui.lineEdit_7.text()),int(self.ui.lineEdit_8.text())))
        cv.imshow("RotationScaling",self.img)
    def Shearing(self):
        p1 = float(self.ui.lineEdit.text())
        p2 = float(self.ui.lineEdit_2.text())
        p3 = float(self.ui.lineEdit_3.text())
        p4 = float(self.ui.lineEdit_4.text())
        p5 = float(self.ui.lineEdit_5.text())
        p6 = float(self.ui.lineEdit_6.text())
        np1 = float(self.ui.lineEdit_9.text())
        np2 = float(self.ui.lineEdit_10.text())
        np3 = float(self.ui.lineEdit_11.text())
        np4 = float(self.ui.lineEdit_12.text())
        np5 = float(self.ui.lineEdit_13.text())
        np6 = float(self.ui.lineEdit_14.text())
        old = np.float32([[p1,p2],
                          [p3,p4],
                          [p5,p6]])
        new = np.float32([[np1,np2],
                          [np3,np4],
                          [np5,np6]])
        M = cv.getAffineTransform(old,new)
        self.img = cv.warpAffine(self.img,M,(int(self.ui.lineEdit_7.text()),int(self.ui.lineEdit_8.text())))
        cv.imshow("Shearing",self.img)
    # 5. Training Cifar10 Classifier Using VGG16
    def ShowTrainImage(self):
        fig = plt.figure()
        for i in range(0,3):
            for j in range(1,4):
                p1 = fig.add_subplot(3,3,i*3+j)
                imgplot = plt.imshow(self.x_train[self.count])
                p1.set_title(self.label[self.y_train[self.count].astype(int)][0])
                self.count = self.count + 1
                plt.axis('off')
        plt.show()
    def ShowHyperparameter(self):
        print("hyperparameters:\n")
        print("batch size: %d\n"%(self.batch_size))
        print("learning rate: %d\n"%(self.learning_rate))
        print("optimizer: %s\n"%(self.optimizer))
    def ShowModelStructure(self):
        self.model.summary()
    def Train(self):
        model = tf.keras.Sequential(name="MyVGG16")

        model.add(Conv2D(32,(3,3),activation='relu',padding='same',input_shape=(32,32,3)))
        model.add(Conv2D(32,(3,3),activation='relu',padding='same'))
        model.add(MaxPooling2D((2,2)))

        model.add(Conv2D(64,(3,3),activation='relu',padding='same'))
        model.add(Conv2D(64,(3,3),activation='relu',padding='same'))
        model.add(MaxPooling2D((2,2)))

        model.add(Flatten())
        model.add(Dense(512,activation='relu'))
        model.add(Dense(10,activation='softmax'))

        model.summary()
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),loss=["sparse_categorical_crossentropy" ],metrics=['accuracy'])


        history = model.fit(self.x_train,self.y_train,epochs = 1,batch_size=128,validation_data = (self.x_test,self.y_test))

        score = model.evaluate(self.x_test,self.y_test,batch_size=128)
        print("訓練分數:",score)

        result = model.predict(self.x_test,batch_size=128)
        print("原始直:",np.squeeze(self.y_test[:20],axis=1))
        print("預測值:",np.argmax(result[:20],axis=1))

        plt.plot(history.history['accuracy'], label='accuracy')
        plt.plot(history.history['val_accuracy'], label='val_accuracy')
        plt.plot(history.history['loss'], label='loss')
        plt.plot(history.history['val_loss'], label='val_loss')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.ylim([0,2])
        plt.legend(loc='lower right')
        plt.show()
        import os
        savedir = "./model"
        model_name = "MyVGG16.h5"
        if not os.path.isdir(savedir):
            os.makedirs(savedir)
        model_path = os.path.join(savedir, model_name)
        model.save(model_path)
    def Predict(self):
        self.model = tf.keras.models.load_model('./model/MyVGG16.h5')
        data = np.reshape(self.x_test[int(self.ui.lineEdit_15.text())],(1,32,32,3))
        result = self.model.predict(data,batch_size=128)
        print(result)
        print("原始直:",self.y_test[int(self.ui.lineEdit_15.text())])
        print("預測值:",np.argmax(result,axis=1))


  

    def open_file(self):
        filename, filetype = QFileDialog.getOpenFileName(self,"Open folder","./")
        return filename
    def update(self,x):
        self.value = x


