
import logging as log
import os
import sys
from io import BytesIO
from PyQt5 import QtWidgets
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QFileDialog, QMessageBox
import warnings
import matplotlib.image as mpimg
from matplotlib import pyplot as plt

import cv2
import numpy as np

warnings.filterwarnings("error")
log.basicConfig(filename='mainLogs.log', filemode='w', format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')
from gui import Ui_MainWindow
class ApplicationWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(ApplicationWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.ui.freq_spinbox.setValue(50)


        self.ui.actionOpen.triggered.connect(lambda: open_file())
        self.ui.actionExport.triggered.connect(lambda: export_file())
        self.ui.filter_combobox.currentIndexChanged.connect(lambda: filters())
        self.ui.kernel_spinbox.valueChanged.connect(lambda:filters())
        self.ui.freq_spinbox.valueChanged.connect(lambda:filters())
        self.ui.equalize_btn.clicked.connect(lambda:equlize())

        original_data = {}
        edited_data = {}


        def open_file():
            filename = QFileDialog.getOpenFileName(filter="png(*.png)")
            if filename[1] != "png(*.png)":
                error = QMessageBox()
                error.setIcon(QMessageBox.Critical)
                error.setWindowTitle("File format Error!")
                error.setText("Please choose a .png file!")
                error.exec_()
            else:
                log.warning("img path = " + filename[0])
                img = cv2.imread(filename[0],0)

                #getting fourier of an 2d array
                f = np.fft.fft2(img)

                #Shift fourier to make center in the middle
                fshift = np.fft.fftshift(f)

                #calculate the magnitude
                fourier_mag = 20 * np.log(np.abs(fshift))

                update_pixmap(img,self.ui.o_img_label)
                update_pixmap(fourier_mag, self.ui.o_fourier_label)
                update_histogram(get_histogram_freq(img),self.ui.o_histogram_label)

                original_data["img"]=img
                original_data["fourier"]=fourier_mag
                original_data["sfft"]=fshift



        #update the image in the gui
        def update_pixmap(img,loc):
            output = BytesIO()
            mpimg.imsave(output, img,cmap="gray")
            output.seek(0)
            pixmap = QPixmap()
            pixmap.loadFromData(output.read())
            loc.setPixmap(pixmap)


        #save the edited img as png
        def export_file():
            img= edited_data["img"]
            filename = QFileDialog.getSaveFileName(self, 'export png', "Image.png",
                                                   'Png files(.png);;All files()')
            cv2.imwrite(filename[0], img)
            log.warning("Img exported at = " + filename[0])

        #main filters function, Check the combo box index to know which filter
        #then calls the filter function, which then return the edited 2d image
        #the edited data carry the result of filter and its fourier
        #then update the edited panel (img/fourier/histogram)
        def filters():
            value=self.ui.filter_combobox.currentIndex()
            result=original_data["img"]
            if(value==0):
                result=lowpass_filter()
            elif(value==1):
                result=highpass_filter()
            elif(value==2):
                result=median_blur()
            else:
                result=laplace_filter()
            fft_img=get_fft(result)
            edited_data["img"] = result
            edited_data["fourier"] = fft_img
            update_pixmap(result, self.ui.e_img_label)
            update_pixmap(fft_img, self.ui.e_fourier_label)
            update_histogram(get_histogram_freq(result), self.ui.e_histogram_label )







        # blur,the central element of the image is replaced by the median of all the pixels in the kernel area.
        def median_blur():
            kernel_size = self.ui.kernel_spinbox.value()
            log.warning("Median blur with ksize = " + str(kernel_size))
            result = cv2.medianBlur(original_data["img"], ksize=kernel_size)
            return result

        #edge detection
        #getting radius from combobox
        #make the mask = 255-mask, because mask radius is from origin which has low frequency
        #blur the mask
        #calculate fft of the mask then multiplay fft of mask by the fft of img
        #normailze fft then transform it back
        def highpass_filter():
            radius=self.ui.freq_spinbox.value()
            mask = 255-getmask(radius)
            #img,ksize,sigma --> Gaussian kernel standard deviation
            mask = cv2.GaussianBlur(mask, (19, 19), 0)

            fshift = original_data["sfft"]
            dft_shift_masked = np.multiply(fshift, mask) / 255
            back_ishift_masked = np.fft.ifftshift(dft_shift_masked)
            img_filtered = np.fft.ifft2(back_ishift_masked, axes=(0, 1))
            result = np.abs(img_filtered).clip(0, 255).astype(np.uint8)

            log.warning("High pass with radius = " + str(radius))
            #img=original_data["img"]
            #result= img - ndimage.gaussian_filter(img, sigma)
            return result

        #blur
        #getting radius from combobox
        #make the mask
        #blur the mask
        #calculate fft of the mask then multiplay fft of mask by the fft of img
        #normailze fft then transform it back
        def lowpass_filter():
            radius=self.ui.freq_spinbox.value()
            mask=getmask(radius)
            # img,ksize,sigma --> Gaussian kernel standard deviation
            mask = cv2.GaussianBlur(mask, (19, 19), 0)
            fshift=original_data["sfft"]
            dft_shift_masked = np.multiply(fshift, mask) / 255
            back_ishift_masked = np.fft.ifftshift(dft_shift_masked)
            img_filtered = np.fft.ifft2(back_ishift_masked, axes=(0, 1))
            result = np.abs(img_filtered).clip(0, 255).astype(np.uint8)

            log.warning("Low pass with radius = " + str(radius))
            #img = original_data["img"]
            #result = ndimage.gaussian_filter(img, sigma)

            return result

        #used for edge detection
        def laplace_filter():
            kernel_size=self.ui.kernel_spinbox.value()
            log.warning("Laplace filter with ksize = " + str(kernel_size))
            result = cv2.Laplacian(original_data["img"], cv2.CV_32F, ksize=kernel_size)
            return result

        #calculate mag_spectrum of fourier of 2darray
        def get_fft(img):
            f = np.fft.fft2(img)
            fshift = np.fft.fftshift(f)
            mag_spectrum = 20 * np.log(np.abs(fshift))
            return mag_spectrum

        #calculate histogram of img , by looping over the 2d array and increase the histogram index
        def get_histogram_freq(img):
            values = np.zeros(256,dtype=int)

            cv2.imwrite("temp.png",img)
            img = cv2.imread("temp.png", 0)

            for i in range(img.shape[0]):
                for j in range(img.shape[1]):
                    index = img[i][j]
                    values[index] += 1
            os.remove("temp.png")
            return values


        #calculate cdf which is used in equalization
        #first fill 256 array with zeros
        #first cdf = first histogram bin
        # then cdf = prev cdf + histogram bin
        # after that normalize cdf by divide over the maximum cdf which is last one cdf[-1] or cdf[255],
        #then multiplay it by 255 since the img is from 0-255
        def get_cdf(hist):
            cdf = np.zeros(256)
            cdf[0] = hist[0]
            for i in range(1, 256):
                cdf[i] = cdf[i - 1] + hist[i]

            #normalize
            cdf = np.round(np.multiply(cdf, 255 / cdf[-1]))
            return cdf


        #calculate histogram
        #calculate cdf
        #loop over the img and change the pixel value to crossponding cdf
        def equalize_image(img):
            hist = get_histogram_freq(img)
            cdf = np.round(get_cdf(hist), 0)
            for i in range(img.shape[0]):
                for j in range(img.shape[1]):
                    img[i][j] = cdf[img[i][j]]
            return img


        #equlaize the img
        #calculate fft of img
        #save img&fft
        #display it in edited panel
        def equlize():
            img=original_data["img"]
            eq_img=equalize_image(img)
            fft_img = get_fft(eq_img)
            edited_data["img"] = eq_img
            edited_data["fourier"] = fft_img
            update_pixmap(fft_img, self.ui.e_fourier_label)
            update_pixmap(eq_img,self.ui.e_img_label)
            update_histogram(get_histogram_freq(eq_img),self.ui.e_histogram_label )

        #make an array with bins ranges from 0-255
        #display and the histogram in the location(original/edited)
        def update_histogram(hist,loc):
            bins = np.arange(0, 256, 1)
            fig = plt.hist(hist, bins=bins,histtype="bar",color="b")
            plt.title('Mean')
            plt.xlabel("value")
            plt.ylabel("Frequency")
            output = BytesIO()
            plt.savefig(output)
            output.seek(0)
            pixmap = QPixmap()
            pixmap.loadFromData(output.read())
            loc.setPixmap(pixmap)

        def getmask(radius):
            img = original_data["img"]
            mask = np.zeros_like(img)
            cy = mask.shape[0] // 2 #center in y , lenght/2

            cx = mask.shape[1] // 2 #center in x , width/2
            #img, center, radius, color,thickness
            #thickness=-1 means that it is solid cirlce not hollow
            #index 0 is to take only 1st channel
            cv2.circle(mask, (cx, cy), radius, (255, 255, 255), -1)[0]
            return mask




def main():
    app = QtWidgets.QApplication(sys.argv)
    application = ApplicationWindow()
    application.show()
    app.exec_()


if __name__ == "__main__":
    main()
