#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# %load Done01-0114.py
import os,sys
import cv2,csv
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image,ImageDraw
import pytesseract,os
from collections import defaultdict
import time
import math
from openpyxl import Workbook

#if not os.path.isdir("./cut01"):
#       os.mkdir("./cut01") 
# tStart = time.time()
# print("let's start")

#cv2讀圖片，並二值化得adaptive__threshold
def cv2_to_adaptive__threshold(image_file_name):
    image_color = cv2.imread(image_file_name)
    im_h ,im_w,color_channel = image_color.shape  #　01:(856,1231,3)
    image = cv2.cvtColor(image_color, cv2.COLOR_BGR2GRAY)
    adaptive_threshold = cv2.adaptiveThreshold(
        image,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2)
    
    return adaptive_threshold,im_w ,im_h

def Vertical(image, axis=0):
    vertical_sum = np.sum(image, axis=0)
    #plt.plot(range(vertical_sum.shape[0]),vertical_sum)
    #plt.gca()
    #plt.show()
    return vertical_sum
        
def Horizontal(image, axis=1):
    horizontal_sum = np.sum(image, axis=1)
    #plt.plot(horizontal_sum, range(horizontal_sum.shape[0]))
    #plt.gca().invert_yaxis()
    #plt.show()  
    return horizontal_sum

def image_sum(image):
    hori =  np.sum(image, axis=1)
    sum_all =  np.sum(hori, axis=0)
    return sum_all/255

def extract_peek_ranges_from_array(array_vals, minimun_val=20, minimun_range=5):
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
        #else:
        #    raise ValueError("cannot parse this case...")
    return peek_ranges

def cut_bar_then_cut_box(adaptive_threshold,y_minimun_val=20,y_minimun_range=5,x_minimun_val=20,x_minimun_range=5):
    vertical = Vertical(adaptive_threshold/255)
    horizontal= Horizontal(adaptive_threshold/255)
    peek_yranges = extract_peek_ranges_from_array(horizontal, minimun_val=20, minimun_range=5)
    peek_xranges = extract_peek_ranges_from_array(vertical, minimun_val=20, minimun_range=5)
    small_images =[]
    vertical_peek_ranges2d = []
    for peek_range in peek_yranges:
        start_y = peek_range[0]
        end_y = peek_range[1]
        line_img = adaptive_threshold[start_y:end_y, :]
        vertical_sum = Vertical(line_img)
        vertical_peek_ranges = extract_peek_ranges_from_array(vertical_sum,minimun_val=30,minimun_range=3) 
        #vertical_peek_ranges2d.append(vertical_peek_ranges) 
    
    return peek_yranges,vertical_peek_ranges#得所有小圖的陣列

def saperate_by_space(array_vals, space_size, minimun_val):    
    start_i = None
    end_i = None
    peek_ranges = []
    space_counter = 0

    for i, val in enumerate(array_vals):                 
        if val >= minimun_val and start_i is None:
            start_i = i
        elif val >= minimun_val and start_i is not None:
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

def color_image_threshold(im,percent,i,j,sum_value):
    table = []
    gray_image = im.convert('L') #轉為灰度圖
    threshold = np.max(gray_image)*percent
    for k in range(256):
        if k < threshold:
            table.append(0)
        else:
            table.append(1)       
    bin_image = gray_image.point(table,'1')
    
    #如為黑底白字，反轉為白底黑字
    sum_all = image_sum(bin_image)
    #print(str(j)+"_"+str(i) , sum_all)
    if sum_all < sum_value :
        bin_image = bin_image.point(lambda x : 1-x)
    return bin_image


#圖片需為正方形才方便使用函數
def clear_circle(img):
    im_h ,im_w,c = img.shape
    radius = int(im_h/2)
    radius2 = int(im_w/2)
    r = math.ceil(im_h/2)
    r2 = math.ceil(im_w/2)
    img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret,thresh1 = cv2.threshold(img2,127,255,cv2.THRESH_BINARY_INV)
    #thresh1 = thresh1/255 
    black = np.argwhere(thresh1>0)
    
    for k,point in enumerate(black):
        a = point[0]
        b = point[1]
        
        center1 =  np.array([radius,radius2])
        center2 =  np.array([radius,r2])
        center3 =  np.array([r,radius2])
        center4 =  np.array([r,r2])
        #print(np.array([a,b]),center1)
        d1 = np.linalg.norm(point-center1)
        d2 = np.linalg.norm(point-center2)
        d3 = np.linalg.norm(point-center3)
        d4 = np.linalg.norm(point-center4)
        d = int(1/4*(d1+d2+d3+d4))
        
        if d > radius-18:
            p1 = a
            p2 = b
            img[a,b] = 255
            #print(p1,p2)
        ret,thresh1 = cv2.threshold(img,175,255,cv2.THRESH_BINARY)
    return thresh1    
    #plt.imshow(img) 
    #cv2.imwrite(img_name)
    
def rotate_img(image,angle,is_sqare):
    if is_sqare:
        im_h ,im_w = image.size
        if im_h==im_w:
            image = image.rotate(angle)
    else:
        image = image.rotate(angle)
    return image
def add_to_square(img,newsize):
    im_h ,im_w,c = img.shape
    #將圖片擴充成正方形，方便用圓周去點
    u = int((newsize-im_h)/2)
    v = int((newsize-im_w)/2)
    img = cv2.copyMakeBorder(img,u,newsize-im_h-u,v,newsize-im_w-v, cv2.BORDER_CONSTANT,value=[255,255,255])
    return img

def image_name(file_name,index_x,index_y,i,j):
    final = file_name+"/"+str(index_y[j]) + str(index_x[i])+".jpg"
    return final


# In[ ]:


pytesseract.pytesseract.tesseract_cmd = '.\Tesseract-OCR\\tesseract.exe'
image_file_name = "./FPK_07.jpg"
adaptive_threshold,im_w ,im_h = cv2_to_adaptive__threshold(image_file_name)
im = Image.open("./FPK_07.jpg")

index_x = [str(7-a) for a in range(0,7)]
index_x.insert(8,"0") 
index_y = ["B","C","D","R","G","F","J","Y"]
peek_yranges,vertical_peek_ranges2d = cut_bar_then_cut_box(adaptive_threshold)
path="./cut07"
if not os.path.isdir(path):
    os.mkdir(path)
print(peek_yranges)
print(vertical_peek_ranges2d)


# In[ ]:


pytesseract.pytesseract.tesseract_cmd = '.\Tesseract-OCR\\tesseract.exe'
image_file_name = "./FPK_07.jpg"
adaptive_threshold,im_w ,im_h = cv2_to_adaptive__threshold(image_file_name)
im = Image.open("./FPK_07.jpg")

index_x = [str(7-a) for a in range(0,7)]
index_x.insert(8,"0") 
index_y = ["Y","B","C","D","R","G","F","J"]
peek_yranges,vertical_peek_ranges2d = cut_bar_then_cut_box(adaptive_threshold)
path="./cut07"
if not os.path.isdir(path):
    os.mkdir(path)

#cnt = 0
for j, peek_range in enumerate(peek_yranges):
    for i, vertical_range in enumerate(vertical_peek_ranges2d):
        x = vertical_range[0]
        y = peek_range[0]
        x2 = vertical_range[1] 
        y2 = peek_range[1] 
        #pt1 = (x, y)
        #pt2 = (x2, y2)
        #cv2.rectangle(image_color, pt1, pt2, color)
        #cnt +=1
        im2 = im.crop((x,y,x2,y2))
        #彩圖二值化
        im2 = color_image_threshold(im2,65/100,i,j,10*10)
        img_name = image_name(path,index_x,index_y,i,j)
        #img_name2 = path+"/bin"+str(index_y[j]) + str(index_x[i])+".jpg"
        #img_name3 =path+"/r"+str(index_y[j]) + str(index_x[i])+".jpg"
        im2.save(img_name)
        if i==7 or j ==0:
            continue
        bin_image = Image.open(img_name)
        image = bin_image .rotate(30)
        image.save(img_name)
        
        ##開啟要辨識的圖片
        img = cv2.imread(img_name)
        img = add_to_square(img,320)
        im_h ,im_w,c = img.shape
        if im_w == im_h:
            img = clear_circle(img)
            ret,img = cv2.threshold(img,175,255,cv2.THRESH_BINARY)    
        cv2.imwrite(img_name,img)


# In[ ]:



def add_board(name):
    old_im = Image.open(name)
    old_size = old_im.size
    new_size = (130,130)
    new_im = Image.new("RGB", new_size)   ## luckily, this is already black!
    new_im.paste(old_im, (int((new_size[0]-old_size[0])/2),
                  int((new_size[1]-old_size[1])/2)))
    return new_im

def cv2_add_board(name,size):
    img = cv2.imread(name)
    im_h ,im_w ,c= img.shape
    #將圖片擴充成正方形，方便用圓周去點
    u = int((size-im_h)/2)
    v = int((size-im_w)/2)
    img = cv2.copyMakeBorder(img,u,size-im_h-u,v,size-im_w-v, cv2.BORDER_CONSTANT,value=[255,255,255])


def write_xls(ws,row,list_c,word):
    for i in range(len(list_c)):
        number=row+list_c[i]
        ws.append([number,word[i]])
        print(number,word[i])

wb = Workbook()
ws = wb.active
ws.append(["NUMBER","CONTENT"])

list_c=[]
list_r=[]
word=[]

for j, peek_range in enumerate(peek_yranges):
    for i, vertical_range in enumerate(vertical_peek_ranges2d):
        img_name = image_name(path,index_x,index_y,i,j)
        img = cv2.imread(img_name) 
        code=pytesseract.image_to_string(img,lang='eng',config='-psm 7 ')
        code=code.replace("O", "0",3)
        code=code.replace(" ", "",2)
        code=code.replace("S", "5",1)
        if len(code)>1 :
            if code[0]=='0':
                code='O'+code[1:]
        if i<8 and len(code)==1 :
            code=code.replace("]", "7",1)
            code=code.replace("A", "4",1)
        # print(i,code)
        if len(code)==1:

            if code.isdigit()==True:
                list_c.append(code)
            else:
                row=code
                write_xls(ws,row,list_c,word)
                word.clear()
                # list_r.append(code)
        else:
            word.append(code)

        wb.save('Team025_07.xlsx')



        # wb = Workbook()
        # ws = wb.active
        # ws.append(["NUMBER","CONTENT"])
        # for i in range(1,64):
        #     name='./01f/'+str(i)+'.jpg'
        #     img = cv2.imread(name) 
        #     code=pytesseract.image_to_string(img,lang='eng',config='-psm 7 ')
        #     code=code.replace("O", "0",3)
        #     code=code.replace(" ", "",2)
        #     code=code.replace("S", "5",1)
        #     if len(code)>1 :
        #         if code[0]=='0':
        #             code='O'+code[1:]
        #     if i<8 and len(code)==1 :
        #         code=code.replace("]", "7",1)
        #         code=code.replace("A", "4",1)
        #     print(i,code)
        #     ws.append([i,code])
        # wb.save('FPK_01.xlsx')
            


        # In[ ]:




