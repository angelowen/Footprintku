#!/usr/bin/env python
# coding: utf-8
import os,sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image,ImageDraw
import pytesseract,os
from collections import defaultdict
import csv

image_color = cv2.imread("./FPK_01.jpg")
im_h ,im_w,color_channel = image_color.shape  #　01:(856,1231,3)
image = cv2.cvtColor(image_color, cv2.COLOR_BGR2GRAY)
im = Image.open("./FPK_01.jpg")
adaptive_threshold = cv2.adaptiveThreshold(
    image,
    255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY_INV, 11, 2)

vertical_sum = np.sum(adaptive_threshold, axis=0)
vertical = vertical_sum/255
horizontal_sum = np.sum(adaptive_threshold, axis=1)
horizontal= horizontal_sum/255

a = np.where(vertical >(im_h/3))
b = np.where(horizontal >(im_w/3))
A = np.array(a)
B = np.array(b)
AA = np.size(A)
BB = np.size(B)
# print (a,b,A,B,AA,BB)

box_y = np.where(vertical > (im_h/2)) #400大約為圖像高度的一半
box_x = np.where(horizontal > (im_w/2))
#消去最外框線
# for x in range(0,x-1): 
for j in box_y:
    adaptive_threshold[:,j] = 0
for i in box_x:
    adaptive_threshold[i,:]=0
# cv2.imshow('line image',adaptive_threshold)
cv2.waitKey(0)
cv2.destroyAllWindows()    

horizontal_sum = np.sum(adaptive_threshold, axis=1)
vertical_sum = np.sum(adaptive_threshold, axis=0)
def extract_peek_ranges_from_array(array_vals, minimun_val=200, minimun_range=2):
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


line_seg_adaptive_threshold = np.copy(adaptive_threshold)
peek_ranges = extract_peek_ranges_from_array(horizontal_sum)
# print("peek:",peek_ranges)
# print("line",line_seg_adaptive_threshold)
for i, peek_range in enumerate(peek_ranges):
    x = 0
    y = peek_range[0]
    w = line_seg_adaptive_threshold.shape[1]
    h = peek_range[1] - y
    pt1 = (x, y)
    pt2 = (x + w, y + h)
    # print(pt1 ,";", pt2,"\n")
    #cv2.rectangle(影像, 頂點座標, 對向頂點座標, 顏色, 線條寬度)
    c = cv2.rectangle(line_seg_adaptive_threshold, pt1, pt2, (0,255,0))
    #wp = pt1 + pt2
    #print(wp)
# cv2.imshow('line image',line_seg_adaptive_threshold)
cv2.waitKey(0)
cv2.destroyAllWindows()

vertical_peek_ranges2d = []
for peek_range in peek_ranges:
    start_y = peek_range[0]
    end_y = peek_range[1]
    line_img = adaptive_threshold[start_y:end_y, :]
    vertical_sum = np.sum(line_img, axis=0)
    for i in box_x:
        horizontal_sum[i] = 0
    vertical_peek_ranges = extract_peek_ranges_from_array(
        vertical_sum,
        minimun_val=50,
        minimun_range=3) #07 OK
    vertical_peek_ranges2d.append(vertical_peek_ranges) 

## Draw
color = (0, 0, 255)
tup =()
cor_y = ()
cnt = 0

path="./cut01"
if not os.path.isdir(path):
    os.mkdir(path)
#二值化圖片的空矩陣
for i, peek_range in enumerate(peek_ranges):
    for j, vertical_range in enumerate(vertical_peek_ranges2d[i]):
        x = vertical_range[0]
        y = peek_range[0]
        w = vertical_range[1] - x
        h = peek_range[1] - y
        pt1 = (x, y)
        pt2 = (x + w, y + h)
        cv2.rectangle(image_color, pt1, pt2, color)
        table = [] 
        cnt +=1
        im2 = im.crop((x,y,x+w,y+h))
        grey_image = im2.convert('L') #轉為灰度圖
        threshold = np.max(grey_image)*0.55
        for k in range(256):
            if k < threshold:
                table.append(0)
            else:
                table.append(1)        
        bin_image = grey_image.point(table,'1')
        bin_image.save("./cut01/" +str(cnt) +".jpg")
#########################################################
def getPixel(image,x,y,G,N):
    L = image.getpixel((x,y))
    if L > G:
        L = True
    else:
        L = False
 
    nearDots = 0
    if L == (image.getpixel((x - 1,y - 1)) > G):
        nearDots += 1
    if L == (image.getpixel((x - 1,y)) > G):
        nearDots += 1
    if L == (image.getpixel((x - 1,y + 1)) > G):
        nearDots += 1
    if L == (image.getpixel((x,y - 1)) > G):
        nearDots += 1
    if L == (image.getpixel((x,y + 1)) > G):
        nearDots += 1
    if L == (image.getpixel((x + 1,y - 1)) > G):
        nearDots += 1
    if L == (image.getpixel((x + 1,y)) > G):
        nearDots += 1
    if L == (image.getpixel((x + 1,y + 1)) > G):
        nearDots += 1
 
    if nearDots < N:
        return image.getpixel((x,y-1))
    else:
        return None

# 降噪 
# 根据一个点A的RGB值，与周围的8个点的RBG值比较，设定一个值N（0 <N <8），当A的RGB值与周围8个点的RGB相等数小于N时，此点为噪点 
# G: Integer 图像二值化阀值 
# N: Integer 降噪率 0 <N <8 
# Z: Integer 降噪次数 
# 输出 
#  0：降噪成功 
#  1：降噪失败 
def clearNoise(image,G,N,Z):
    draw = ImageDraw.Draw(image)
 
    for i in range(0,Z):
        for x in range(1,image.size[0] - 1):
            for y in range(1,image.size[1] - 1):
                color = getPixel(image,x,y,G,N)
                if color != None:
                    draw.point((x,y),color)

def convert_Image(image, standard=190):
    '''
    【灰度轉換】
    '''
    image = image.convert('L')

    clearNoise(image,10,1,4)
    '''
    
    【二值化】
    根據閾值 standard , 將所有像素都置為 0(黑色) 或 255(白色), 便於接下來的分割
    '''
    pixels = image.load()
    for x in range(image.width):
        for y in range(image.height):
            if pixels[x, y] > standard:
                pixels[x, y] = 255
            else:
                pixels[x, y] = 0
    return image
# path="./crop"
# if not os.path.isdir(path):
#     os.mkdir(path)
with open('FPK_01.csv','w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["NUMBER","CONTENT"])
    cnt=0
    for i in range(12,22):#12,72
        dis=15
        cnt+=1
        name="./cut01/"+str(i)+".jpg"##開啟要辨識的圖片
        with Image.open(name) as img:
            im_h ,im_w = img.size
            img=convert_Image(img)
            if im_w>2*dis and im_h>2*dis:
                if i<21:
                    img = img.crop((0+dis, 0+dis*2, im_w-dis*2, im_h -dis))  # (left, upper, right, lower)
            out=img.resize((img.size[0]*2, img.size[1]*2),Image.ANTIALIAS).save("./crop/" +str(i) +".jpg")##放大後辨識率較佳
            # img.show()
            img1=Image.open("./crop/" +str(i) +".jpg")
            code=pytesseract.image_to_string(img1,lang="eng")#,lang='eng',config='-psm 7 digits'
            if cnt%10==0:
                pass
            else:
                print('A'+str(cnt%10),code)#,"ans:",lines[j]
                writer.writerow(['A'+str(cnt%10),code])
            

    f_n=65
    cnt=0
    for i in range(22,72):#12,72
        dis=15
        name="./cut01/"+str(i)+".jpg"##開啟要辨識的圖片
        with Image.open(name) as img:
            im_h ,im_w = img.size
            img=convert_Image(img)
            if im_w>2*dis and im_h>2*dis:
                if i<21:
                    img = img.crop((0+dis, 0+dis*2, im_w-dis*2, im_h -dis))  # (left, upper, right, lower)
                else:
                    img = img.crop((0+dis, 0+dis*2, im_w-dis*1.3, im_h -dis))  # (left, upper, right, lower)
            out=img.resize((img.size[0]*2, img.size[1]*2),Image.ANTIALIAS).save("./crop/" +str(i) +".jpg")##放大後辨識率較佳
            # img.show()
            img1=Image.open("./crop/" +str(i) +".jpg")
            code=pytesseract.image_to_string(img1,lang="eng")#,lang='eng',config='-psm 7 digits'
            if cnt%10==0:
                f_n+=1
                pass
            else:
                print(chr(f_n)+str(cnt%10),code)#,"ans:",lines[j]
                writer.writerow([chr(f_n)+str(cnt%10),code])
            cnt+=1
