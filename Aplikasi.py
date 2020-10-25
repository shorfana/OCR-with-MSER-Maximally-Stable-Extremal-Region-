# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'OCR.ui'
#
# Created by: PyQt5 UI code generator 5.12.3
#
# WARNING! All changes made in this file will be lost!


from __future__ import with_statement
import cv2 
import numpy as np 
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QTabWidget
from PyQt5.QtWidgets import QDialog, QFileDialog
from PyQt5.QtGui import QIcon, QPixmap, QImage
from PyQt5.QtWidgets import QWidget, QPushButton, QLineEdit, QInputDialog, QApplication, QLabel, QMessageBox
import sys
from skimage.transform import rescale, resize


class Ui_OCR(object):
    def setupUi(self, OCR):
        OCR.setObjectName("OCR")
        OCR.resize(1360, 695)
        self.Qore = QtWidgets.QTabWidget(OCR)
        self.Qore.setGeometry(QtCore.QRect(0, 0, 1361, 691))
        self.Qore.setObjectName("Qore")
        self.tabBeranda = QtWidgets.QWidget()
        self.tabBeranda.setObjectName("tabBeranda")
        self.haloberanda = QtWidgets.QLabel(self.tabBeranda)
        self.haloberanda.setGeometry(QtCore.QRect(10, 20, 211, 16))
        self.haloberanda.setObjectName("haloberanda")
        self.Qore.addTab(self.tabBeranda, "")
        self.tabPengolahanData = QtWidgets.QWidget()
        self.tabPengolahanData.setObjectName("tabPengolahanData")
        self.framePengolahanData = QtWidgets.QFrame(self.tabPengolahanData)
        self.framePengolahanData.setGeometry(QtCore.QRect(0, 0, 1351, 661))
        self.framePengolahanData.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.framePengolahanData.setFrameShadow(QtWidgets.QFrame.Raised)
        self.framePengolahanData.setObjectName("framePengolahanData")
        self.groupBoxGambar = QtWidgets.QGroupBox(self.framePengolahanData)
        self.groupBoxGambar.setGeometry(QtCore.QRect(149, 40, 571, 531))
        self.groupBoxGambar.setObjectName("groupBoxGambar")
        self.labelGambarPreviewPengolahanData = QtWidgets.QLabel(self.groupBoxGambar)
        self.labelGambarPreviewPengolahanData.setGeometry(QtCore.QRect(6, 49, 561, 451))
        self.labelGambarPreviewPengolahanData.setObjectName("labelGambarPreviewPengolahanData")
        self.groupBoxHasil = QtWidgets.QGroupBox(self.framePengolahanData)
        self.groupBoxHasil.setGeometry(QtCore.QRect(740, 40, 571, 531))
        self.groupBoxHasil.setObjectName("groupBoxHasil")
        self.lblPreviewGambarHasil = QtWidgets.QLabel(self.groupBoxHasil)
        self.lblPreviewGambarHasil.setGeometry(QtCore.QRect(0, 50, 561, 451))
        self.lblPreviewGambarHasil.setObjectName("lblPreviewGambarHasil")
        self.groupBoxMenu = QtWidgets.QGroupBox(self.framePengolahanData)
        self.groupBoxMenu.setGeometry(QtCore.QRect(10, 40, 120, 191))
        self.groupBoxMenu.setObjectName("groupBoxMenu")
        self.btnPilihFilePD = QtWidgets.QPushButton(self.groupBoxMenu)
        self.btnPilihFilePD.setGeometry(QtCore.QRect(10, 30, 75, 23))
        self.btnPilihFilePD.setObjectName("btnPilihFilePD")
        self.btnGrayscale = QtWidgets.QPushButton(self.groupBoxMenu)
        self.btnGrayscale.setGeometry(QtCore.QRect(10, 70, 75, 23))
        self.btnGrayscale.setObjectName("btnGrayscale")
        self.btnMSER = QtWidgets.QPushButton(self.groupBoxMenu)
        self.btnMSER.setGeometry(QtCore.QRect(10, 110, 75, 23))
        self.btnMSER.setObjectName("btnMSER")
        self.btnPotong = QtWidgets.QPushButton(self.groupBoxMenu)
        self.btnPotong.setGeometry(QtCore.QRect(10, 150, 75, 23))
        self.btnPotong.setObjectName("btnPotong")
        self.Qore.addTab(self.tabPengolahanData, "")
        self.tabPelatihan = QtWidgets.QWidget()
        self.tabPelatihan.setObjectName("tabPelatihan")
        self.groupBoxPelatihan = QtWidgets.QGroupBox(self.tabPelatihan)
        self.groupBoxPelatihan.setGeometry(QtCore.QRect(0, 10, 1351, 651))
        self.groupBoxPelatihan.setObjectName("groupBoxPelatihan")
        self.btnProsesPelatihan = QtWidgets.QPushButton(self.groupBoxPelatihan)
        self.btnProsesPelatihan.setGeometry(QtCore.QRect(10, 20, 75, 23))
        self.btnProsesPelatihan.setObjectName("btnProsesPelatihan")
        self.framePelatihan = QtWidgets.QFrame(self.groupBoxPelatihan)
        self.framePelatihan.setGeometry(QtCore.QRect(10, 50, 1331, 601))
        self.framePelatihan.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.framePelatihan.setFrameShadow(QtWidgets.QFrame.Raised)
        self.framePelatihan.setObjectName("framePelatihan")
        self.Qore.addTab(self.tabPelatihan, "")
        self.tabPengujian = QtWidgets.QWidget()
        self.tabPengujian.setObjectName("tabPengujian")
        self.framePengujian = QtWidgets.QFrame(self.tabPengujian)
        self.framePengujian.setGeometry(QtCore.QRect(0, 0, 1351, 661))
        self.framePengujian.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.framePengujian.setFrameShadow(QtWidgets.QFrame.Raised)
        self.framePengujian.setObjectName("framePengujian")
        self.groupBoxHasilPengujian = QtWidgets.QGroupBox(self.framePengujian)
        self.groupBoxHasilPengujian.setGeometry(QtCore.QRect(741, 10, 571, 531))
        self.groupBoxHasilPengujian.setObjectName("groupBoxHasilPengujian")
        self.groupBoxGambar_2 = QtWidgets.QGroupBox(self.framePengujian)
        self.groupBoxGambar_2.setGeometry(QtCore.QRect(150, 10, 571, 531))
        self.groupBoxGambar_2.setObjectName("groupBoxGambar_2")
        self.lblPreviewGambarUji = QtWidgets.QLabel(self.groupBoxGambar_2)
        self.lblPreviewGambarUji.setGeometry(QtCore.QRect(6, 49, 561, 451))
        self.lblPreviewGambarUji.setObjectName("lblPreviewGambarUji")
        self.groupBoxMenu_2 = QtWidgets.QGroupBox(self.framePengujian)
        self.groupBoxMenu_2.setGeometry(QtCore.QRect(0, 10, 120, 111))
        self.groupBoxMenu_2.setObjectName("groupBoxMenu_2")
        self.btnPilihFileUji = QtWidgets.QPushButton(self.groupBoxMenu_2)
        self.btnPilihFileUji.setGeometry(QtCore.QRect(10, 30, 75, 23))
        self.btnPilihFileUji.setObjectName("btnPilihFileUji")
        self.btnProsesUji = QtWidgets.QPushButton(self.groupBoxMenu_2)
        self.btnProsesUji.setGeometry(QtCore.QRect(10, 70, 75, 23))
        self.btnProsesUji.setObjectName("btnProsesUji")
        self.Qore.addTab(self.tabPengujian, "")

        self.retranslateUi(OCR)
        self.Qore.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(OCR)

    def retranslateUi(self, OCR):
        _translate = QtCore.QCoreApplication.translate
        OCR.setWindowTitle(_translate("OCR", "Optical Character Recognition (MSER)"))
        self.haloberanda.setText(_translate("OCR", "Halo ini beranda"))
        self.Qore.setTabText(self.Qore.indexOf(self.tabBeranda), _translate("OCR", "Beranda"))
        self.groupBoxGambar.setTitle(_translate("OCR", "Gambar"))
        self.labelGambarPreviewPengolahanData.setText(_translate("OCR", "Preview"))
        self.groupBoxHasil.setTitle(_translate("OCR", "Hasil"))
        self.lblPreviewGambarHasil.setText(_translate("OCR", "Preview Hasil"))
        self.groupBoxMenu.setTitle(_translate("OCR", "Menu"))
        self.btnPilihFilePD.setText(_translate("OCR", "Pilih File"))
        self.btnGrayscale.setText(_translate("OCR", "Grayscale"))
        self.btnMSER.setText(_translate("OCR", "MSER"))
        self.btnPotong.setText(_translate("OCR", "Potong"))
        self.Qore.setTabText(self.Qore.indexOf(self.tabPengolahanData), _translate("OCR", "Pengolahan Data"))
        self.groupBoxPelatihan.setTitle(_translate("OCR", "Pelatihan"))
        self.btnProsesPelatihan.setText(_translate("OCR", "Proses"))
        self.Qore.setTabText(self.Qore.indexOf(self.tabPelatihan), _translate("OCR", "Pelatihan"))
        self.groupBoxHasilPengujian.setTitle(_translate("OCR", "Hasil"))
        self.groupBoxGambar_2.setTitle(_translate("OCR", "Gambar"))
        self.lblPreviewGambarUji.setText(_translate("OCR", "Preview"))
        self.groupBoxMenu_2.setTitle(_translate("OCR", "Menu"))
        self.btnPilihFileUji.setText(_translate("OCR", "Pilih File"))
        self.btnProsesUji.setText(_translate("OCR", "Proses"))
        self.Qore.setTabText(self.Qore.indexOf(self.tabPengujian), _translate("OCR", "Pengujian"))

        self.btnPilihFilePD.clicked.connect(self.clickFile)
        self.btnGrayscale.clicked.connect(self.grayProses)
        self.btnMSER.clicked.connect(self.MSERProsesTrain)
        self.btnPotong.clicked.connect(self.CharCut)


    def clickFile(self):
        global fileName
        fileName, _ = QtWidgets.QFileDialog.getOpenFileName(None, "Select Image", "", "Image File (*.png *.jpg *.jpeg)")
        if fileName:
            self.labelGambarPreviewPengolahanData.setText(fileName)
            pixmap = QtGui.QPixmap(fileName)
            pixmap = pixmap.scaled(self.labelGambarPreviewPengolahanData.width(), self.labelGambarPreviewPengolahanData.height(), QtCore.Qt.KeepAspectRatio)
            self.labelGambarPreviewPengolahanData.setPixmap(pixmap)
            self.labelGambarPreviewPengolahanData.setAlignment(QtCore.Qt.AlignCenter)    

    def grayProses(self):
        image = cv2.imread(fileName)
        grayImage = PreproTrain.grayscale(self,image)  

        cv2.imwrite("pengolahan_data/Grayscale.jpg", grayImage)
        pixmap = QtGui.QPixmap("pengolahan_data/Grayscale.jpg")
        pixmap = pixmap.scaled(self.lblPreviewGambarHasil.width(), self.lblPreviewGambarHasil.height(), QtCore.Qt.KeepAspectRatio)
        self.lblPreviewGambarHasil.setPixmap(pixmap)
        self.lblPreviewGambarHasil.setAlignment(QtCore.Qt.AlignCenter)    

    def MSERProsesTrain(self):
        image = cv2.imread(fileName)
        grayImage = PreproTrain.grayscale(self,image) 
        mserDetection = PreproTrain.mserTextDetection(self,grayImage) 

        cv2.imwrite("pengolahan_data/mserdetect.jpg", mserDetection)
        pixmap = QtGui.QPixmap("pengolahan_data/mserdetect.jpg")
        pixmap = pixmap.scaled(self.lblPreviewGambarHasil.width(), self.lblPreviewGambarHasil.height(), QtCore.Qt.KeepAspectRatio)
        self.lblPreviewGambarHasil.setPixmap(pixmap)
        self.lblPreviewGambarHasil.setAlignment(QtCore.Qt.AlignCenter)  

    def CharCut(self):
        image = cv2.imread(fileName)
        grayImage = PreproTrain.grayscale(self,image) 
        mserDetection = PreproTrain.cutSegment(self,grayImage)  


class PreproTrain(QWidget):
    def __init__(self,parent=None):
        super(PreproTrain, self).__init__(parent)

        self.textDelta = QLineEdit(self)
        self.textDelta.move(100,22)
        self.textDelta.setPlaceholderText("Masukan Nilai Delta")

        self.setGeometry(300,300,290,140)
        self.setWindowTitle("Input Inisialisasi")

    def grayscale(self,image):
        grayValue = 0.1140 * image[:,:,2] + 0.5870 * image[:,:,1] + 0.2989 * image[:,:,0]
        gray_img = grayValue.astype(np.uint8)
        return gray_img    


    def mserTextDetection(self,grayImage):
        global txtDelta, txtMinA, txtMaxA, txtMaxV
        delta, result = QInputDialog.getText(None, 'Peringatan!', 'Harap masukan inisialisasi parameter Delta')
        MinArea, result = QInputDialog.getText(None, 'Peringatan!', 'Harap masukan inisialisasi parameter Min Aera')
        MaxArea, result = QInputDialog.getText(None, 'Peringatan!', 'Harap masukan inisialisasi parameter Max Area')
        MaxVariation, result = QInputDialog.getText(None, 'Peringatan!', 'Harap masukan inisialisasi parameter Max Variation')
        txtDelta = int(delta)
        txtMinA = int(MinArea)
        txtMaxA = int(MaxArea)
        txtMaxV = float(MaxVariation)
        mser = cv2.MSER_create(_delta = txtDelta,_min_area = txtMinA ,_max_area = txtMaxA ,_max_variation = txtMaxV)##0.0689
        #detect regions in gray scale image
        # vis = grayImage.copy()    
        regions, _ = mser.detectRegions(grayImage)
        hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]
        mask = np.zeros((grayImage.shape[0], grayImage.shape[1], 1), dtype=np.uint8)
        # nmss = nms(regions,hulls )
        # i=0 
        # keep = []
        for contour in hulls:
            x,y,w,h = cv2.boundingRect(contour)
            # cv2.rectangle(vis, (x, y), (x + w, y + h), (255, 255, 255), 1)
            # cv2.imwrite('data_segmentasi/{}.png'.format(i), grayImage[y:y+h,x:x+w])
            #     i = i+1
            cv2.drawContours(mask, [contour], -1, (255, 255, 255), -1)
        segmen_result_textonly = cv2.bitwise_and(grayImage, grayImage, mask=mask)

        # cv2.imshow("hulls", vis)
        return segmen_result_textonly 

    def cutSegment(self,grayImage):
        
        mser = cv2.MSER_create(_delta = txtDelta,_min_area = txtMinA ,_max_area = txtMaxA ,_max_variation = txtMaxV)##0.0689
        #detect regions in gray scale image
        regions, _ = mser.detectRegions(grayImage)
        hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]
        mask = np.zeros((grayImage.shape[0], grayImage.shape[1], 1), dtype=np.uint8)
        # nmss = nms(regions,hulls )
        i=0 
        # keep = []
        for contour in hulls:
            x,y,w,h = cv2.boundingRect(contour)
            # cv2.rectangle(vis, (x, y), (x + w, y + h), (255, 255, 255), 1)
            # cv2.imwrite('data_segmentasi/{}.png'.format(i), grayImage[y:y+h,x:x+w])
            # thresholding = grayImage[y:y+h,x:x+w]
            # h,w = np.shape(thresholding)
            # #alur threshold
            # for x in range(h):
            #     for y in range(w):
            #         if (thresholding[x][y] > 195):
            #             thresholding[x][y] = 0
            #         else:
            #             thresholding[x][y] = 255 
            cv2.imshow('data segmentasi', grayImage[y:y+h,x:x+w] )
            text, result = QInputDialog.getText(None, 'Peringatan!', 'Harap masukan label pada karakter yang sudah dipotong')
            cv2.imwrite('data_segmentasi/{}.png'.format(text), grayImage[y:y+h,x:x+w])
            cv2.waitKey()    
            i = i+1

     



if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Dialog = QtWidgets.QDialog()
    ui = Ui_OCR()
    ui.setupUi(Dialog)
    Dialog.show()
    sys.exit(app.exec_())



 # def pengolahanData(self):
    #     image = cv2.imread(fileName)
    #     grayImage = PreproTrain.grayscale(self,image)
    #     mserDetection = PreproTrain.mserTextDetection( self,grayImage)  

    #     cv2.imwrite("pengolahan_data/Grayscale.jpg", grayImage)
    #     pixmap = QtGui.QPixmap("pengolahan_data/Grayscale.jpg")
    #     pixmap = pixmap.scaled(self.lblHasilGrayscale.width(), self.lblHasilGrayscale.height(), QtCore.Qt.KeepAspectRatio)
    #     self.lblHasilGrayscale.setPixmap(pixmap)
    #     self.lblHasilGrayscale.setAlignment(QtCore.Qt.AlignCenter)

    #     cv2.imwrite("pengolahan_data/mserdetect.jpg", mserDetection)
    #     pixmap = QtGui.QPixmap("pengolahan_data/mserdetect.jpg")
    #     pixmap = pixmap.scaled(self.lblHasilMSER.width(), self.lblHasilMSER.height(), QtCore.Qt.KeepAspectRatio)
    #     self.lblHasilMSER.setPixmap(pixmap)
    #     self.lblHasilMSER.setAlignment(QtCore.Qt.AlignCenter)     