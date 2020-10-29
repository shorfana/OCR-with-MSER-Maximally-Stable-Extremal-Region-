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
import os
import math 
import csv
# from PIL import Image, ImageFilter
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler, MinMaxScaler


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
        self.Nama = QtWidgets.QLabel(self.tabBeranda)
        self.Nama.setGeometry(QtCore.QRect(40, 40, 911, 16))
        self.Nama.setObjectName("Nama")
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
        self.groupBoxMenu.setGeometry(QtCore.QRect(10, 40, 120, 231))
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
        self.btnEkstaksi = QtWidgets.QPushButton(self.groupBoxMenu)
        self.btnEkstaksi.setGeometry(QtCore.QRect(10, 190, 75, 23))
        self.btnEkstaksi.setObjectName("btnEkstaksi")
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
        self.Nama.setText(_translate("OCR", "Nama Muhammad Iqbal Shorfana"))
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
        self.btnEkstaksi.setText(_translate("OCR", "Ekstraksi Fitur"))
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
        self.btnEkstaksi.clicked.connect(self.ekstraksi)
        self.btnProsesPelatihan.clicked.connect(self.SVMPelatihan)
        self.btnPilihFileUji.clicked.connect(self.SVMClicked)
        self.btnProsesUji.clicked.connect(self.SVMPengujian)

    

    def clickFile(self):
        global fileName, fileNama
        fileName, _ = QtWidgets.QFileDialog.getOpenFileName(None, "Select Image", "", "Image File (*.png *.jpg *.jpeg)")
        if fileName:
            base = os.path.basename(fileName)
            split = os.path.splitext(base)
            fileNama = os.path.splitext(base)[0]
            print(fileNama)
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
        mserDetection = PreproTrain.cutSegment(self,grayImage,fileNama)  

    def ekstraksi(self):
        ekstraksi = PreproTrain.zoningExtraction(self,fileNama)    

    def SVMPelatihan(self):
        global namaFile
        namaFile, _ = QtWidgets.QFileDialog.getOpenFileName(None, "Select File", "", "CSV File (*.csv )")
        if namaFile:
            pelatihan = SVMTrain.trainData(self,namaFile)

    def SVMClicked(self):
        global filePengujian
        filePengujian, _ = QtWidgets.QFileDialog.getOpenFileName(None, "Select Image", "", "Image File (*.png *.jpg *.jpeg)")
        if filePengujian:
            self.lblPreviewGambarUji.setText(filePengujian)
            pixmap = QtGui.QPixmap(filePengujian)
            pixmap = pixmap.scaled(self.lblPreviewGambarUji.width(), self.lblPreviewGambarUji.height(), QtCore.Qt.KeepAspectRatio)
            self.lblPreviewGambarUji.setPixmap(pixmap)
            self.lblPreviewGambarUji.setAlignment(QtCore.Qt.AlignCenter)   

    def SVMPengujian(self):
        image = cv2.imread(filePengujian)
        grayImage = PreproTest.grayscale(self,image)   
        mserDetection = PreproTest.MSERProsesTest(self, grayImage)




class PreproTrain(QWidget):
    def __init__(self,parent=None):
        super(PreproTrain, self).__init__(parent)

    def grayscale(self,image):
        grayValue = 0.1140 * image[:,:,2] + 0.5870 * image[:,:,1] + 0.2989 * image[:,:,0]
        gray_img = grayValue.astype(np.uint8)
        return gray_img    

    def non_max_suppression_fast(boxes, overlapThresh):
        # Empty array detection
        if len(boxes) == 0:
            return []

            # Convert the type to float
        if boxes.dtype.kind == "i":
            boxes = boxes.astype("float")

        pick = []

        # Four coordinate arrays
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        area = (x2 - x1 + 1) * (y2 - y1 + 1)  # Calculate area array
        idxs = np.argsort(y2)  # Returns the index value of the lower right corner coordinate from small to large

        # Start traversing to delete duplicate boxes
        while len(idxs) > 0:
            # Put the bottom right box into the pick array
            last = len(idxs) - 1
            i = idxs[last]
            pick.append(i)

            # Find the largest coordinate x1y1 and the smallest coordinate x2y2 in the remaining boxes,
            xx1 = np.maximum(x1[i], x1[idxs[:last]])
            yy1 = np.maximum(y1[i], y1[idxs[:last]])
            xx2 = np.minimum(x2[i], x2[idxs[:last]])
            yy2 = np.minimum(y2[i], y2[idxs[:last]])

            # Calculate the proportion of overlapping area in the corresponding frame
            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)
            overlap = (w * h) / area[idxs[:last]]

            # If the proportion is greater than the threshold, delete
            idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0])))

        return boxes[pick].astype("int")

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
        vis = grayImage.copy() 
        orig = grayImage.copy()   
        #detect regions in grayscale image
        regions, _ = mser.detectRegions(grayImage)
        hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]
        mask = np.zeros((grayImage.shape[0], grayImage.shape[1], 1), dtype=np.uint8)
        keep = []
        for contour in hulls:
            x,y,w,h = cv2.boundingRect(contour)
            cv2.drawContours(mask, [contour], -1, (255, 255, 255), -1)
        segmen_result_textonly = cv2.bitwise_and(grayImage, grayImage, mask=mask)

        return segmen_result_textonly 

    def cutSegment(self,grayImage,fileNama):
        # dir = QFileDialog.getExistingDirectory(self, tr("Open Directory"),"/home", QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks)
        # print(dir)
        path = 'data_segmentasi/{}'.format(fileNama)
        os.makedirs(path)
        mser = cv2.MSER_create(_delta = txtDelta,_min_area = txtMinA ,_max_area = txtMaxA ,_max_variation = txtMaxV)##0.0689
        vis = grayImage.copy() 
        orig = grayImage.copy() 
        #detect regions in gray scale image
        regions, _ = mser.detectRegions(grayImage)
        hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]
        mask = np.zeros((grayImage.shape[0], grayImage.shape[1], 1), dtype=np.uint8)
        
        keep = []
        for contour in hulls:
            x,y,w,h = cv2.boundingRect(contour)
            keep.append([x, y, x + w, y + h])
            cv2.rectangle(vis, (x, y), (x + w, y + h), (255, 255, 0), 1)

        # Filter for non-repeated rectangular boxes
        keep2 = np.array(keep)
        pick = PreproTrain.non_max_suppression_fast(keep2, 0.5)   
        iterasi = 0
        for (startX, startY, endX, endY) in pick:
            thres1 = orig[startY:endY,startX:endX]
            i, j = np.shape(thres1)
            for a in range(i):
                for b in range(j):
                    if (thres1[a][b] > 165):
                        thres1[a][b] = 0
                    else:
                        thres1[a][b] = 255
            size = (100,100)            
            thres2 = cv2.resize(thres1,size)              
            # cv2.imshow('data segmentasi', thres2)
            # text, result = QInputDialog.getText(None, 'Peringatan!', 'Harap masukan label pada karakter yang sudah dipotong')
            cv2.imwrite('data_segmentasi/{}/{}.png'.format(fileNama,iterasi), thres2)
            cv2.rectangle(orig, (startX, startY), (endX, endY), (255, 185, 120), 2) 
            iterasi = iterasi + 1
        self.msgBox = QMessageBox()
        self.msgBox.setIcon(QMessageBox.Question)
        self.msgBox.setText("Proses Pemotongan Karakter Selesai")
        self.msgBox.setWindowTitle("Pesan")
        self.msgBox.setStandardButtons(QMessageBox.Ok)
        self.msgBox.exec()   

    def zoningExtraction(self,fileNama):
        # ====================  function untuk zooning  ===================================
        # mengambil seluru file dari folder data_segmentasi dan folder font yang dituju
        fileList = []
        myDir = "data_segmentasi/{}".format(fileNama) # lokasi folder
        format = ".png" # ekstensi file yang diambil
        print("Open directory = ",myDir)
        for root, dirs, files in os.walk(myDir, topdown=False):
            for name in files:
                if name.endswith(format):
                    fullName = os.path.join(root, name)
                    fileList.append(fullName)

        # kemudian lakukan perulangan sebanyak file
        for file in fileList:
            # membuka file gambar
            # img_file = Image.open(file)
            gambar = cv2.imread(file)

            # mendapatkan parameter gambar...
            # width, height = img_file.size
            # format = img_file.format
            # mode   = img_file.mode
            # file_name = img_file.filename             

            thresholding = PreproTrain.grayscale(self,gambar)
            h,w = np.shape(thresholding)
            #alur threshold
            for x in range(h):
                for y in range(w):
                    if (thresholding[x][y] > 165):
                        thresholding[x][y] = 1
                    else:
                        thresholding[x][y] = 0 
            size = (21,21)            
            value = cv2.resize(thresholding,size)               
            np.savetxt('matriks global thres.txt',np.array(value),fmt="%s")  
            # cv2.imwrite('gambar.jpg', value2)         

            # mencari nilai centroid
            width, height = value.shape
            r=0
            xc_up = 0
            xy_up = 0
            tot_p = 0
            while r < height:
                c=0
                while c < width:
                    temp1 = r * value[r,c]
                    temp2 = c * value[r,c]
                    xc_up = xc_up + temp1
                    xy_up = xy_up + temp2
                    if value[r,c] == 1:
                        tot_p = tot_p + 1
                    c=c+1
                r=r+1
            # print(xc_up)
            # print(xy_up)
            # print(tot_p)
            xc = xc_up / tot_p
            xy = xy_up / tot_p
            print ("nilai xc =", round(xc))
            print ("nilai xy =", round(xy))
            xc = round(xc)
            xy = round(xy)

            # menghitung zona 1
            total_jarak_zona1 = 0
            banyak_titik = 0
            r=0
            while r < 6:
                c=0
                while c < width:
                    if value[r,c] == 1:
                        total_jarak_zona1 = total_jarak_zona1 + (math.sqrt((r + xc) ** 2 + (c + xy) ** 2))
                        banyak_titik = banyak_titik + 1
                    c=c+1
                r=r+1
            result_zona1 = total_jarak_zona1 / banyak_titik
            print(total_jarak_zona1, banyak_titik)
            print("Zona 1 =", result_zona1)

            # menghitung zona 2
            total_jarak_zona2 = 0
            banyak_titik = 0
            r=6
            while r < 13:
                c=0
                while c < width:
                    if value[r,c] == 1:
                        total_jarak_zona2 = total_jarak_zona2 + (math.sqrt((r + xc) ** 2 + (c + xy) ** 2))
                        banyak_titik = banyak_titik + 1
                    c=c+1
                r=r+1
            result_zona2 = total_jarak_zona2 / banyak_titik
            print(total_jarak_zona2, banyak_titik)
            print("Zona 2 =", result_zona2)

            # menghitung zona 3
            total_jarak_zona3 = 0
            banyak_titik = 0
            r=13
            while r < 20:
                c=0
                while c < width:
                    if value[r,c] == 1:
                        total_jarak_zona3 = total_jarak_zona3 + (math.sqrt((r + xc) ** 2 + (c + xy) ** 2))
                        banyak_titik = banyak_titik + 1
                    c=c+1
                r=r+1
            result_zona3 = total_jarak_zona3 / banyak_titik
            print(total_jarak_zona3, banyak_titik)
            print("Zona 3 =", result_zona3)

            cv2.imshow('Karakter yang akan diekstraksi',gambar)
            # cv2.imwrite('gambar.jpg', value3)   
            label, result = QInputDialog.getText(None, 'Peringatan!', 'Harap masukan label pada data ekstraksi')
            result = [result_zona1, result_zona2, result_zona3, label]
            # print(result)
            with open("hasil ekstraksi.csv", 'a') as f:
                writer = csv.writer(f)
                writer.writerow(result)

        self.msgBox = QMessageBox()
        self.msgBox.setIcon(QMessageBox.Question)
        self.msgBox.setText("Proses Ekstraksi dan Pelabelan Selesai")
        self.msgBox.setWindowTitle("Pesan")
        self.msgBox.setStandardButtons(QMessageBox.Ok)
        self.msgBox.exec()  
        cv2.destroyAllWindows()   
        # ===================  end function untuk zooning  =========================================        


class SVMTrain(QWidget):
    def __init__(self,parent=None):
        super(SVMTrain, self).__init__(parent)


    def trainData(self,data):
        global svm_clf, sc, data_x, data_y

        # Baca data
        datacsv = pd.read_csv(data)
        dataraw = np.array(datacsv)

        # Pilih data x dan y
        data_x = dataraw[:, 0:3]
        data_y = dataraw[:, 3]

        # Normalisasi data menggunakan MinMax
        sc = MinMaxScaler()
        data_x = sc.fit_transform(data_x)

        #Lakukan pemilihan data train dan test
        # x_train,x_test, y_train, y_test  = train_test_split(data_x, data_y, test_size=0.5, random_state=10)
        # print("X train",x_train)
        # print("X test",x_test)
        # print("Y train",y_train)
        # print("Y test",y_test)

        # Model SVM
        svm_clf = svm.SVC(kernel='rbf')

        # pelatihan
        svm_clf.fit(data_x, data_y)
        print("X train",data_x)
        print("Y train",data_y)

        self.msgBox = QMessageBox()
        self.msgBox.setIcon(QMessageBox.Question)
        self.msgBox.setText("Pelatihan SVM selesai")
        self.msgBox.setWindowTitle("Pesan")
        self.msgBox.setStandardButtons(QMessageBox.Ok)
        self.msgBox.exec()  

    def testData(self,data):
        # global svm_clf, sc
        
        data =  sc.transform(data)
        hasil = svm_clf.predict(data) 
        print('data normalisasi ',data)  

        kelas = hasil[0]
        print(kelas)
        return kelas     
        


class PreproTest(QWidget):
    def __init__(self,parent=None):
        super(PreproTest, self).__init__(parent)

    def grayscale(self,image):
        grayValue = 0.1140 * image[:,:,2] + 0.5870 * image[:,:,1] + 0.2989 * image[:,:,0]
        gray_img = grayValue.astype(np.uint8)
        return gray_img 

    def zoningTest(self,value):
        width, height = value.shape
        r=0
        xc_up = 0
        xy_up = 0
        tot_p = 0
        while r < height:
            c=0
            while c < width:
                temp1 = r * value[r,c]
                temp2 = c * value[r,c]
                xc_up = xc_up + temp1
                xy_up = xy_up + temp2
                if value[r,c] == 1:
                    tot_p = tot_p + 1
                c=c+1
            r=r+1
        # print(xc_up)
        # print(xy_up)
        # print(tot_p)
        xc = xc_up / tot_p
        xy = xy_up / tot_p
        print ("nilai xc =", round(xc))
        print ("nilai xy =", round(xy))
        xc = round(xc)
        xy = round(xy)

        # menghitung zona 1
        total_jarak_zona1 = 0
        banyak_titik = 0
        r=0
        while r < 6:
            c=0
            while c < width:
                if value[r,c] == 1:
                    total_jarak_zona1 = total_jarak_zona1 + (math.sqrt((r + xc) ** 2 + (c + xy) ** 2))
                    banyak_titik = banyak_titik + 1
                c=c+1
            r=r+1
        result_zona1 = total_jarak_zona1 / banyak_titik
        print(total_jarak_zona1, banyak_titik)
        print("Zona 1 =", result_zona1)

        # menghitung zona 2
        total_jarak_zona2 = 0
        banyak_titik = 0
        r=6
        while r < 13:
            c=0
            while c < width:
                if value[r,c] == 1:
                    total_jarak_zona2 = total_jarak_zona2 + (math.sqrt((r + xc) ** 2 + (c + xy) ** 2))
                    banyak_titik = banyak_titik + 1
                c=c+1
            r=r+1
        result_zona2 = total_jarak_zona2 / banyak_titik
        print(total_jarak_zona2, banyak_titik)
        print("Zona 2 =", result_zona2)

        # menghitung zona 3
        total_jarak_zona3 = 0
        banyak_titik = 0
        r=13
        while r < 20:
            c=0
            while c < width:
                if value[r,c] == 1:
                    total_jarak_zona3 = total_jarak_zona3 + (math.sqrt((r + xc) ** 2 + (c + xy) ** 2))
                    banyak_titik = banyak_titik + 1
                c=c+1
            r=r+1
        result_zona3 = total_jarak_zona3 / banyak_titik
        print(total_jarak_zona3, banyak_titik)
        print("Zona 3 =", result_zona3)

        # cv2.imshow('Karakter yang akan diekstraksi',gambar)
        # # cv2.imwrite('gambar.jpg', value3)   
        # label, result = QInputDialog.getText(None, 'Peringatan!', 'Harap masukan label pada data ekstraksi')
        result = [[result_zona1, result_zona2, result_zona3]]
        # print(result)
        # with open("hasil ekstraksi.csv", 'a') as f:
        #     writer = csv.writer(f)
        #     writer.writerow(result)
        return result  

    def MSERProsesTest(self,grayImage):
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
        vis = grayImage.copy() 
        orig = grayImage.copy() 
        #detect regions in gray scale image
        regions, _ = mser.detectRegions(grayImage)
        hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]
        mask = np.zeros((grayImage.shape[0], grayImage.shape[1], 1), dtype=np.uint8)
        
        keep = []
        for contour in hulls:
            x,y,w,h = cv2.boundingRect(contour)
            keep.append([x, y, x + w, y + h])
            cv2.rectangle(vis, (x, y), (x + w, y + h), (255, 255, 0), 1)  

        # Filter for non-repeated rectangular boxes
        keep2 = np.array(keep)
        pick = PreproTrain.non_max_suppression_fast(keep2, 0.5)   
        iterasi = 0
        for (startX, startY, endX, endY) in pick:
            thres1 = orig[startY:endY,startX:endX]
            i, j = np.shape(thres1)
            for a in range(i):
                for b in range(j):
                    if (thres1[a][b] > 165):
                        thres1[a][b] = 0
                    else:
                        thres1[a][b] = 1
            size = (21,21)            
            value = cv2.resize(thres1,size)  
            data = PreproTest.zoningTest(self,value) 
            # print("Ini data pertama",data)
            # data1 = data.reshape(-1,1)
            # print(data)
            klasifikasi = SVMTrain.testData(self,data)         
            # cv2.imshow('data segmentasi', thres2)
            # text, result = QInputDialog.getText(None, 'Peringatan!', 'Harap masukan label pada karakter yang sudah dipotong')
            # cv2.imwrite('data_segmentasi/{}/{}.png'.format(fileNama,iterasi), thres2)
            cv2.rectangle(orig, (startX, startY), (endX, endY), (255, 185, 120), 2) 
            iterasi = iterasi + 1
            with open("hasil klasifikasi.csv", 'a') as f:
                writer = csv.writer(f)
                writer.writerow(klasifikasi)
        self.msgBox = QMessageBox()
        self.msgBox.setIcon(QMessageBox.Question)
        self.msgBox.setText("Klasifikasi selesai")
        self.msgBox.setWindowTitle("Pesan")
        self.msgBox.setStandardButtons(QMessageBox.Ok)
        self.msgBox.exec()   

# class SVMTest(QWidget):
#     def __init__(self,parent=None):
#         super(SVMTrain, self).__init__(parent)

#     def testData(self,data):
#         global svm_clf, sc

#         data =  sc.transform(data)
#         hasil = svm_clf.predict(data)   

#         kelas = hasil[0]

#         return kelas         

     
if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Dialog = QtWidgets.QDialog()
    ui = Ui_OCR()
    ui.setupUi(Dialog)
    Dialog.show()
    sys.exit(app.exec_())
 