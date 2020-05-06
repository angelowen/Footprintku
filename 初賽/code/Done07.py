#!/usr/bin/env python
# coding: utf-8
import csv
import numpy as np
import pytesseract,os
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image,ImageDraw
import matplotlib.pyplot as plt
from collections import defaultdict
from PIL import Image

image_color = cv2.imread("./FPK_07.jpg")
im_h ,im_w,color_channel = image_color.shape 
image = cv2.cvtColor(image_color, cv2.COLOR_BGR2GRAY)
adaptive_threshold = cv2.adaptiveThreshold(
    image,
    255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY_INV, 11, 2)
im = Image.open("./FPK_07.jpg")


vertical_sum = np.sum(adaptive_threshold, axis=0)
vertical = vertical_sum/255

horizontal_sum = np.sum(adaptive_threshold, axis=1)
horizontal= horizontal_sum/255
#使用plt.plot畫(x ,y)曲線


def box_cut(left, upper, right, lower):
    region = im.crop((left, upper, right, lower))
    return region

index_x=['1','2','3','inexy','7','8','9']
index_y = ['A','B','C','D','E','F','G','H','J','K','L','M','N','P','R','T']


box_x = np.where(vertical > (im_h/3)) #400大約為圖像高度的一半
box_y = np.where(horizontal > (im_w/3))
Box_x = box_x[0]
listx = Box_x.tolist()
Box_y = box_y[0]
listy = Box_y.tolist()

for k in range(len(listx)-1):
    if listx[k+1]-listx[k]==1:
        listx[k]=0
for k in range(len(listy)-1):    
    if listy[k+1]-listy[k]==1:
        listy[k]=0

while 0 in listx:        
    listx.remove(0)
while 0 in listy:
    listy.remove(0)
    
print(len(listx))
print(len(listy))


path = "./cut07/"
if not os.path.isdir(path):
    os.mkdir(path)
    
for j in range(len(listy)-1):
    for i in range(len(listx)-1):
        new = box_cut(listx[i],listy[j],listx[i+1],listy[j+1])
        
        new.save(path  +str(index_y[j]) +str(index_x[i]) +".jpg")

path = "./cut07/"
dirs = os.listdir( path )
with open('FPK_07.csv','w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["NUMBER","CONTENT"])
    for filename in dirs:

        img = cv2.imread(path+filename)
        im_h,im_w,tmp=img.shape
        img = img[int(im_h*0.1):im_h-5, int(im_h*0.1):im_w-5]
        # a = cv2.copyMakeBorder(img,5,5,5,5, cv2.BORDER_CONSTANT,value=[255,255,255])
        im_h,im_w,tmp=img.shape
        # print(a.shape)
        a=cv2.resize(img,(im_w*5,im_h*5))
        # cv2.imshow("img",a)
        # cv2.waitKey(0)
        code=pytesseract.image_to_string(a,lang='eng',config='-psm 7 digits')#,lang='eng',config='-psm 7 digits'
        code=code.replace(" ","",2)
        code=code.replace("MI","MJ",2)
        if len(code)>0 and code[len(code)-1]=='O':
                code=code[:len(code)-1]+'0'
        if len(code)>3 and code[:3]=='QQU' and code[3:4]=='S':
                code=code[:len(code)-1]+'5'
        if len(code)>3 and code[:3]=='QQU' and code[3:4]=='Z':
                code=code[:len(code)-1]+'7'
        name = [  str(x) for x in filename.split( '.' )]
        print(name[0],code)
        writer.writerow([name[0],code.encode("utf8").decode("cp950", "ignore")])
        