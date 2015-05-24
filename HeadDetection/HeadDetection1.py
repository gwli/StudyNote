# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

from __future__ import division
import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
import os
import argparse
from IPython.display import  Image

# <codecell>

img = cv2.imread('InClass.png')
img = cv2.cvtColor(img, cv2.cv.CV_BGR2GRAY)
img = cv2.GaussianBlur(img,(3,3),0) 
edges = cv2.Canny(img, 50, 150)  
cv2.imwrite('edges.jpg',edges)
lines = cv2.HoughLinesP(edges,1,np.pi/180, 100, minLineLength= 20, maxLineGap = 10)

# <codecell>

Image('InClass.png')

# <codecell>

cropImg = img
kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
erodedImg = cv2.erode(cropImg,kernel) 
height,width = cropImg.shape
tmp =[]
size = 15
for i in range(1,height-size):
    for jj in range(1,width-size):
        if (1.5*erodedImg[i:i+size,jj]<np.mean(erodedImg)).all() and (1.5*erodedImg[i,jj:jj+size]<np.mean(erodedImg)).all():
            tmp.append([i,jj])
            cv2.circle(img,(jj+np.int(size/2),i+np.int(size/2)),np.int(size/2),(255,0,0))
print img 
cv2.imwrite('detectedPoints.jpg',img)
tmp = np.array(tmp)

# <codecell>

Image('detectedPoints.jpg')

# <codecell>

def blockDetection1():                           
        Num = tmp.shape[0]  
        i,j = 0,1
        
        f = open('out.txt','w')
        while i<Num: 
            oneclass = []
            anotherClass = []
            while j <Num:
                if np.abs(tmp[i,0] - tmp[j,0] )<9 and np.abs(tmp[i,1] - tmp[j,1] )<9:
                    oneclass.append([tmp[j,0],tmp[j,1]])
                else:
                    anotherClass.append([tmp[j,0],tmp[j,1]])
                j += 1
            else:
                oneclass.append([tmp[i,0],tmp[i,1]])
                oneclass = np.mean(np.array(oneclass),0)
                if self.Y<self.height/2.0:
                    locationY = oneclass[0]+self.upY+self.lightPoints
                else:
                    locationY = oneclass[0]+self.upY
                locationX = oneclass[1]+self.leftX  
                theta1 = -self.theta    
                rotateMatrix = [[np.cos(theta1),-np.sin(theta1)],[np.sin(theta1),np.cos(theta1)]]
                pointLocation = np.dot(np.array([locationX-self.X,locationY-self.Y]), rotateMatrix)+np.array([self.X,self.Y])
                print >>f, '%d,%d'%(int(pointLocation[0]),int(pointLocation[1])) 
                cv2.circle(self.img,(int(pointLocation[0]),int(pointLocation[1])),10,(255,0,0))                
                tmp  = np.array(anotherClass)
                Num = tmp.shape[0]  
                i,j = 0,1
                
        cv2.imwrite('detectedPoints.jpg',self.img)

# <codecell>

 def blockDetection(self):   
        self.cropImg = self.img
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
        erodedImg = cv2.erode(self.cropImg,kernel) 
        height,width = self.cropImg.shape
        tmp =[]
        size = 15
        for i in range(1,height-size):
            for jj in range(1,width-size):
                if (1.5*erodedImg[i:i+size,jj]<np.mean(erodedImg)).all() and (1.5*erodedImg[i,jj:jj+size]<np.mean(erodedImg)).all():
                    tmp.append([i,jj])
                    cv2.circle(self.img,(jj+np.int(size/2),i+np.int(size/2)),np.int(size/2),(255,0,0))
        print self.img         
        cv2.imwrite('detectedPoints.jpg',self.img)
        tmp = np.array(tmp) 
        
    def blockDetection1(self):                           
        Num = tmp.shape[0]  
        i,j = 0,1
        
        f = open('out.txt','w')
        while i<Num: 
            oneclass = []
            anotherClass = []
            while j <Num:
                if np.abs(tmp[i,0] - tmp[j,0] )<9 and np.abs(tmp[i,1] - tmp[j,1] )<9:
                    oneclass.append([tmp[j,0],tmp[j,1]])
                else:
                    anotherClass.append([tmp[j,0],tmp[j,1]])
                j += 1
            else:
                oneclass.append([tmp[i,0],tmp[i,1]])
                oneclass = np.mean(np.array(oneclass),0)
                if self.Y<self.height/2.0:
                    locationY = oneclass[0]+self.upY+self.lightPoints
                else:
                    locationY = oneclass[0]+self.upY
                locationX = oneclass[1]+self.leftX  
                theta1 = -self.theta    
                rotateMatrix = [[np.cos(theta1),-np.sin(theta1)],[np.sin(theta1),np.cos(theta1)]]
                pointLocation = np.dot(np.array([locationX-self.X,locationY-self.Y]), rotateMatrix)+np.array([self.X,self.Y])
                print >>f, '%d,%d'%(int(pointLocation[0]),int(pointLocation[1])) 
                cv2.circle(self.img,(int(pointLocation[0]),int(pointLocation[1])),10,(255,0,0))                
                tmp  = np.array(anotherClass)
                Num = tmp.shape[0]  
                i,j = 0,1
                
        cv2.imwrite('detectedPoints.jpg',self.img)

# <codecell>


