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

image_color = cv2.imread("./FPK_10.jpg")
im_h ,im_w,color_channel = image_color.shape  #　01:(856,1231,3)
image = cv2.cvtColor(image_color, cv2.COLOR_BGR2GRAY)
im = Image.open("./FPK_10.jpg")

adaptive_threshold = cv2.adaptiveThreshold(
    image,
    255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY_INV, 11, 2)

def Vertical(image, axis=0):
    vertical_sum = np.sum(image, axis=0)
    return vertical_sum
        
def Horizontal(image, axis=1):
    horizontal_sum = np.sum(image, axis=1)
    return horizontal_sum

def extract_peek_ranges_from_array(array_vals, minimun_val, minimun_range):
    start_i = None
    end_i = None
    peek_ranges = []
    #enumerate() 函數用於將數據對象組合為一個索引序列，同時列出數據和數據下標
    for i, val in enumerate(array_vals):
        if val > minimun_val and start_i is None:
            start_i = i
        elif val > minimun_val and start_i is not None:
            pass
        elif val < minimun_val and start_i is not None:
            end_i = i
            if end_i - start_i >= minimun_range:
                peek_ranges.append((start_i, end_i))
            start_i = None
            end_i = None
        elif val < minimun_val and start_i is None:
            pass
        else:
            raise ValueError("cannot parse this case...")
    return peek_ranges

def saperate_by_space(array_vals, space_size, minimun_val):    
    start_i = None
    end_i = None
    peek_ranges = []
    space_counter = 0

    for i, val in enumerate(array_vals):                 
        if val > minimun_val and start_i is None:
            start_i = i
        elif val > minimun_val and start_i is not None:
            space_counter = 0
            if i is len(array_vals) -1:
                peek_ranges.append((start_i, i))
            end_i = None

        elif val < minimun_val and start_i is not None:
            if space_counter is 0:
                end_i = i;
            space_counter = space_counter + 1

            
            if space_counter is space_size :
                peek_ranges.append((start_i, end_i))
                start_i = None
                end_i = None
        elif val < minimun_val and start_i is None:
            pass
        else:
            raise ValueError("cannot parse this case...")
    return peek_ranges


vertical = Vertical(adaptive_threshold/255)
horizontal = Horizontal(adaptive_threshold/255)

box_y = np.where(vertical > 4*im_h/5)
box_x = np.where(horizontal > 4*im_w/5)

for j in box_y:
    adaptive_threshold[:,j] = 0
for i in box_x:
    adaptive_threshold[i,:]=0


index_x=[]
for i in range(0,14+1):
    index_x.append(str(i))
index_y = ['num','P','N','M','L','K','J','H','G','F','E','D','C','B','A']


horizontal_sum = Horizontal(adaptive_threshold/255)
vertical_sum =  Vertical(adaptive_threshold/255)
peek_yranges = saperate_by_space(horizontal_sum, space_size=5, minimun_val=10) #extract_peek_ranges_from_array(horizontal_sum,)
print(peek_yranges)
peek_ranges = extract_peek_ranges_from_array(vertical_sum,minimun_val=20, minimun_range=3)#saperate_by_space(vertical_sum, space_size=2, minimun_val=30)
print(peek_ranges)



path = "./cut10/"
if os.path.isdir(path):
    command = "rmdir /s /q \"%s\""
    command = command % path
    result = os.system(command)
    print(result)
os.mkdir(path)
    
for j, peek_range in enumerate(peek_yranges):
    #vertical_peek_ranges = extract_peek_ranges_from_array(Vertical(peek_range),minimun_val=10,minimun_range=3) 
    y = peek_range [0]
    y2 = peek_range [1]
    for i ,vertical_range in enumerate(peek_ranges):
        x = vertical_range[0]
        x2 = vertical_range[1]
        im2 = im.crop((x,y,x2,y2))
        im2.save(path  +str(index_y[j])+index_x[i] +".jpg")

dirs = os.listdir( path )
with open('FPK_10.csv','w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["NUMBER","CONTENT"])
    for filename in dirs:
        flag=0
        name = [  str(x) for x in filename.split( '.' )]
        try:
            if(name[0][1]=='0'or name[0][0:3]=='num'):
                continue
        except:
            pass
        for i in range(4,12):
            for j in range(4,12):
                if name[0]==str(index_y[i])+str(j):
                    flag=1
                    break
        if(flag==1):
             continue           
                
        flag=0
        img = cv2.imread(path+filename)
        im_h,im_w,tmp=img.shape
        img = img[5:im_h-5, 2:im_w-2]
        # a = cv2.copyMakeBorder(img,5,5,5,5, cv2.BORDER_CONSTANT,value=[255,255,255])
        im_h,im_w,tmp=img.shape
        # print(a.shape)
        try:
            a=cv2.resize(img,(im_w*20,im_h*20))
        except :
            print("n")
        # cv2.imshow("img",a)
        # cv2.waitKey(0)
        code=pytesseract.image_to_string(a,lang='eng',config='-psm 7 digits')#,lang='eng',config='-psm 7 digits'

        
        if len(code)>0 and code[0]=='C':
                code='E'+code[1:]
        for k in range(len(code)):
            if str.islower(code[k])==True:
                code=code[:k]+str.upper(code[k])+code[k+1:]
        for k in range(len(code)):
            if code[k]=='/':
                flag=1
            if code[k]=='I':
                if flag==1:
                    code1=code[:k]+code[k+1:]
                    flag=2
                else:
                    code=code[:k]+'/'+code[k+1:]
                    break
        if(flag==2):
            code=code1
        for k in range(len(code)):
            if code[k]=='('and code[1]!='/':
                code=code[0]+'/'+code[1:]
        code=code.replace("(/","",1)
        code=code.replace("-","",1)
        code=code.replace("y","Y",1)
        code=code.replace("¥","Y",1)
        code=code.replace("//","I/",1)
        code=code.replace("i","/",1)
        code=code.replace("]","I",1)
        if len(code)==2 and code[len(code)-1]=='/':
                code=code[:]+'I'
        
        print(name[0],code)
        writer.writerow([name[0],code.encode("utf8").decode("cp950", "ignore")])