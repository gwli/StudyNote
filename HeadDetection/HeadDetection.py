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

# <codecell>

    def __init__(self, imgpath):
        if os.path.exists(imgpath) == "False":
            return "can't open img :" + imgpath
        self.img = cv2.imread(imgpath)
#        self.img = self.img/255
        self.img = cv2.cvtColor(self.img, cv2.cv.CV_BGR2GRAY)
        self.img = cv2.GaussianBlur(self.img,(3,3),0) 
        self.dirname = os.path.dirname(os.path.abspath(imgpath))

        self.edges = cv2.Canny(self.img, 50, 150)  
        cv2.imwrite('edges.jpg',self.edges)
        self.lines = cv2.HoughLinesP(self.edges,1,np.pi/180, 100, minLineLength= 20, maxLineGap = 10)
        
    def drawline(self):
        for x1,y1,x2,y2 in self.lines[0]:
           cv2.line(self.img,(x1,y1),(x2,y2),(0,255,0),2)
           cv2.imwrite('drawline.jpg',self.img)
        
    def calculateTheta(self):
        LineNum = len(self.lines[0])
        self.HoriLines = []
        VeriLines = []
        self.theta = []
        self.count = 0
        for i in range(LineNum):
            x1,y1,x2,y2 = self.lines[0][i]
            self.tanAngle = (y2-y1)/(np.float(x2-x1))
            if   np.abs(self.tanAngle) <0.04:
               self.HoriLines.append(self.lines[0][i])
               self.count +=1
               self.theta.append(1*math.atan(self.tanAngle))
               self.Y = y1
               self.X = x1
            else:
                 VeriLines.append(self.lines[0][i])
        if self.count ==0:
            print u'Change the coefficient of parameter HoughLinesP'
        self.theta = np.mean(self.theta)
        self.angle = math.degrees(self.theta)
        
        
        ## affine transform
        
    def affineTransform(self):
        self.image_center = (self.X,self.Y)
        rot_mat = cv2.getRotationMatrix2D(self.image_center,self.angle,1)
        self.height,self.width = self.img.shape   
        self.adjustedOriginal = cv2.warpAffine(self.img, rot_mat, (self.width,self.height))
        cv2.imwrite('adjustedOriginal.jpg',self.adjustedOriginal)
        
     ##extract the upy and downy
    def upDownY(self):
        sumValues = np.sum(self.edges/255,axis = 1)
        if self.Y<self.height/2.0:  ##Upper hemisphere 
            self.upY = self.Y
            for i in range(self.height,self.Y,-1):
                if (sumValues[i:i-10:-1]<sumValues.min()+6).all():
                    self.downY  = i
                    break
        else:        ##Lower  hemisphere 
            self.downY = self.Y
            for i in range(1,self.Y):
                if (sumValues[i:i+10]<sumValues.min()+6).all():
                    self.upY = i
                    break           
                
        plt.savefig("hist.jpg",sumValues)   
    def leftRightX(self):
        leftX = []
        rightX = []
        for i in range(self.count):
            leftX.append(self.HoriLines[i][0]) 
            rightX.append(self.HoriLines[i][2]) 
            
        self.leftX = np.min(np.array(leftX))
        self.rightX =  np.max(np.array(rightX))
        
        ##Crop the rectangle runaway
    def cropImage(self):
        adjustedOriginal = cv2.cv.fromarray(self.adjustedOriginal)
        self.lightPoints = 18  # 杩欎釜鏄惁鏈夌敤
        if self.Y<self.height/2.0:  ## Up
            print (self.leftX,self.upY+self.lightPoints,self.rightX-self.leftX,self.downY-self.upY-self.lightPoints)
            self.CropImg = cv2.cv.GetSubRect(adjustedOriginal,(self.leftX,self.upY+self.lightPoints,self.rightX-self.leftX,self.downY-self.upY-self.lightPoints))
        else:     ##Down
            print (self.leftX,self.upY,self.rightX-self.leftX,self.downY-self.upY-self.lightPoints)
            self.CropImg = cv2.cv.GetSubRect(adjustedOriginal,(self.leftX,self.upY,self.rightX-self.leftX,self.downY-self.upY-self.lightPoints))
        
        self.cropImg = np.asarray(self.CropImg)
        cv2.imwrite('cropImg.jpg',self.cropImg)

       ##denoise by using the mask averaging
    def denoising(self):
        k=3
        a = np.ones((k,k),dtype = np.int)     
        mid = (k-1)/2
        for i in range(1):
            a[mid, mid] = i
            kern = a/np.sum(a)
            self.dimg = cv2.filter2D(self.cropImg, -1, kern)
            cv2.imwrite("denoising.jpg",self.dimg) 
            
            
            ##Target detection using simple clustering
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
                if np.abs(tmp[i,0] - tmp[j,0] )<9 or np.abs(tmp[i,1] - tmp[j,1] )<9:
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
        f.close()
    
    def execute(self):
         self.blockDetection()

# <codecell>

if __name__ == '__main__':
   parser = argparse.ArgumentParser(description="ImageAirport Author:zhang gege ")
   parser.add_argument('--img',help='image path', type=str,default='image_data/image_0001.jpg')
   args = parser.parse_args()
   one = ImageAirport(args.img)
   one.execute()

# <codecell>

ls 

# <codecell>


