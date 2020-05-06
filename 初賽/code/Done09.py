import pytesseract,os
import time
from PIL import Image,ImageDraw
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np
import cv2,csv
#!/usr/bin/env python
# coding: utf-8

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

image_color = cv2.imread("./FPK_09.jpg")
im_h ,im_w,color_channel = image_color.shape 
image = cv2.cvtColor(image_color, cv2.COLOR_BGR2GRAY)
adaptive_threshold = cv2.adaptiveThreshold(
    image,
    255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY_INV, 11, 2)
im = Image.open("./FPK_09.jpg")



vertical_sum = np.sum(adaptive_threshold, axis=0)
vertical = vertical_sum/255
horizontal_sum = np.sum(adaptive_threshold, axis=1)
horizontal= horizontal_sum/255
box_x = np.where(vertical > (im_h/2)) #400大約為圖像高度的一半
box_y = np.where(horizontal > (im_w/2))
#去掉大框
for i in box_y:
    adaptive_threshold[i,:] = 0
for j in box_x:
    adaptive_threshold[:,j]=0
    
horizontal_sum = np.sum(adaptive_threshold, axis=1)
horizontal= horizontal_sum/255
#使用plt.plot畫(x ,y)曲線

vertical_sum = np.sum(adaptive_threshold, axis=0)
vertical = vertical_sum/255

Box_x = box_x[0]
listx = Box_x.tolist()
Box_x

Box_y = box_y[0]
listy = Box_y.tolist()
Box_y

def box_cut(left, upper, right, lower):
    region = im.crop((left, upper, right, lower))
    return region

def box_ranges_from_array(lis,a,b):
    if lis[b]-lis[a] > 10:
        return True
    else:
        return False


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
        #else:
        #    raise ValueError("cannot parse this case...")
    return peek_ranges


# In[49]:


A = extract_peek_ranges_from_array(vertical,4,45)
B = extract_peek_ranges_from_array(horizontal,10,22)

path="./cut09"
if not os.path.isdir(path):
    os.mkdir(path)
cnt = 0
for j in range(len(B)):
    for i in range(len(A)):
        x1 = A[i][0]
        y1 = B[j][0]
        x2 = A[i][1] 
        y2 = B[j][1] 
        table = []
        new = box_cut(x1-2,y1-1,x2+1,y2+1) #裁成小圖
        grey_image = new.convert('L') #轉為灰度圖
        threshold = 200
        for k in range(256):
            if k < threshold:
                  table.append(0)
            else:
                table.append(1)
        ##table = find_array_frame(256,threshold,blank_table)
                
        bin_image = grey_image.point(table, '1')
        bin_image.save("./cut09/" +str(cnt) +".jpg")   
        cnt +=1        
###########################################
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

mylist=['A','B','C','D','E','F','G','H']
cnt=0
with open('FPK_09.csv','w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["NUMBER","CONTENT"])
    for i in range(7,37):#12,72
        dis=15
        name="./cut09/"+str(i)+".jpg"##開啟要辨識的圖片
        with Image.open(name) as img:
            im_h ,im_w = img.size
            img1 = cv2.imread(name)
            a = cv2.copyMakeBorder(img1,5,5,5,5, cv2.BORDER_CONSTANT,value=[255,255,255])
            code=pytesseract.image_to_string(a,lang='eng',config='-psm 8 ')#,lang='eng',config='-psm 7 digits'
            w_len=len(code)
            flag=0
            new=""
            start=-1
            end=w_len
            for j in range(0,w_len):
                if str.isupper(code[j])==True:
                    if flag==0:
                        start=j
                        flag=1
                    else:
                        end=j
                if start!=-1 and str.isdigit(code[j])==False and str.isupper(code[j])==False:
                    end=j-1
                    break

            new=code[start:end+1]
            print(mylist[int(cnt/7)]+str(cnt%7+1),new)
            writer.writerow([mylist[int(cnt/7)]+str(cnt%7+1),new])
            cnt+=1

    for i in range(37,63):
        dis=10
        name="./cut09/"+str(i)+".jpg"##開啟要辨識的圖片
        with Image.open(name) as img:
            im_w ,im_h = img.size
            img=convert_Image(img)
            # if i>=43 and i<47:
            #     # img = img.crop((0, 0,im_w,im_h,))  # (left, upper, right, lower)
            #     img1 = cv2.imread(name)
            #     a = cv2.copyMakeBorder(img1,5,5,5,5, cv2.BORDER_CONSTANT,value=[255,255,255])
            #     code=pytesseract.image_to_string(a,lang='eng',config='-psm 8 ')
            # else:
            img.resize((img.size[0]*2, img.size[1]*2),Image.ANTIALIAS).save("./crop/" +str(i) +".jpg")
            img1=Image.open("./crop/" +str(i) +".jpg")
            code=pytesseract.image_to_string(img1,lang='eng',config='-psm 8 ')#,lang='eng',config='-psm 7 digits'
            
            w_len=len(code)
            flag=0
            new=""
            start=-1
            end=w_len
            for j in range(0,w_len):
                if str.isupper(code[j])==True:
                    if flag==0:
                        start=j
                        flag=1
                    else:
                        end=j
                if start!=-1 and str.isdigit(code[j])==False and str.isalpha(code[j])==False:
                    end=j-1
                    break

            new=code[start:end+1]
            print(mylist[int(cnt/7)]+str(cnt%7+1),new)
            writer.writerow([mylist[int(cnt/7)]+str(cnt%7+1),new])
            cnt+=1
            
            
