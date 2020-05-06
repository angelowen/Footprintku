import os,sys,re
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image,ImageDraw
import pytesseract,os
from collections import defaultdict
import csv
import time
from openpyxl import Workbook

# tStart = time.time()
# print("let's start")
"""
img_name = "./FPK_03.jpg"
pytesseract.pytesseract.tesseract_cmd = '.\Tesseract-OCR\\tesseract.exe'
image_color = cv2.imread(img_name)
im_h ,im_w,color_channel = image_color.shape  #　01:(856,1231,3)
image = cv2.cvtColor(image_color, cv2.COLOR_BGR2GRAY)
im = Image.open(img_name)
adaptive_threshold = cv2.adaptiveThreshold(
    image,
    255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY_INV, 11, 2)

name = "./cut03"
if not os.path.isdir(name):
    os.mkdir(name)
"""

def Vertical(image, axis=0):
    vertical_sum = np.sum(image, axis=0)
    # plt.plot(range(vertical_sum.shape[0]),vertical_sum)
    # plt.gca()
    # plt.show()
    return vertical_sum
        
def Horizontal(image, axis=1):
    horizontal_sum = np.sum(image, axis=1)
    # plt.plot(horizontal_sum, range(horizontal_sum.shape[0]))
    # plt.gca().invert_yaxis()
    # plt.show()  
    return horizontal_sum

def box_cut(left, upper, right, lower):
    img_name = "./FPK_03.jpg"
    im = Image.open(img_name)
    region = im.crop((left, upper, right, lower))
    return region

def box_ranges_from_array(lis,a,b):

    if lis[b]-lis[a] == 1:
        return False
    else:
        return True
    
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
        # else:
            # raise ValueError("cannot parse this case...")
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
            if i is len(array_vals) -1:#last one or not
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
        #else:
        #    raise ValueError("cannot parse this case...")
    return peek_ranges


#存完小圖
def cutpic():
    img_name = "./FPK_03.jpg"
    pytesseract.pytesseract.tesseract_cmd = '.\Tesseract-OCR\\tesseract.exe'
    image_color = cv2.imread(img_name)
    im_h ,im_w,color_channel = image_color.shape  #　01:(856,1231,3)
    image = cv2.cvtColor(image_color, cv2.COLOR_BGR2GRAY)
    im = Image.open(img_name)
    adaptive_threshold = cv2.adaptiveThreshold(
        image,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2)

    name = "./cut03"
    if not os.path.isdir(name):
        os.mkdir(name)
    
    vertical = Vertical(adaptive_threshold/255)
    horizontal= Horizontal(adaptive_threshold/255)

    coor1 = saperate_by_space(vertical , space_size=5, minimun_val=3)
    coor2 = saperate_by_space(horizontal , space_size=5, minimun_val=15)
    print(coor1,coor2)

    sx=0
    sy=0

    for i,coorx in enumerate(coor1):
        c1 = coorx[0]
        c2 = coorx[1]
        if c2-c1 < 50:
            sx = c1
    for j,coory in enumerate(coor2):
        c3 = coory[0]
        c4= coory[1]
        if c4-c3 < 50:
            sy = c3 
    print("sx,sy",sx,sy)
    box_x = np.where(vertical > (im_h/2)) 
    box_y = np.where(horizontal > (im_w/2))
    
    Box_x = box_x[0]
    listx = Box_x.tolist()
    #print(Box_x)

    Box_y = box_y[0]
    listy = Box_y.tolist()

    print(listx)    
    print(listy)   

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

    listx.insert(0,sy) 
    listy.insert(0,sx)  

    print("listx",listx)
    Ti = []
    Tj = []
    blank_table = [] #二值化圖片的空矩陣
    plus_cnt=0
    for j in range(int(len(listy))-1):
        if box_ranges_from_array(listy,j,j+1): 
            for i in range(int(len(listx))-1):
                if box_ranges_from_array(listx,i,i+1):
                    table = []
                    if i==0 or j==0:
                        new = box_cut(listx[i]-2,listy[j]+2,listx[i+1]-3,listy[j+1]-3)
                    else:
                        new = box_cut(listx[i]+1,listy[j]+1,listx[i+1]-3,listy[j+1]-3) #裁成小圖
                    grey_image = new.convert('L') #轉為灰度圖
                    #grey_image.save("./cut04/" +str(index_y[i])+str(index_x[j]) +".jpg")

                    width = listx[i+1]-listx[i]-6
                    height = listy[j+1]-listy[j]-6

                    #二值化圖片#有些效果不好
                    threshold = np.max(grey_image)*0.85
                    for k in range(256):
                        if k < threshold:
                            table.append(0)
                        else:
                            table.append(1)
                    ##table = find_array_frame(256,threshold,blank_table)
                    bin_image = grey_image.point(table, '1')
                    
                    bin_image.save("./cut03/" +str(j)+'_'+str(i) +".jpg")
                    


cutpic()
c_l = [str(i) for i in range(62,73)]
c_l.reverse()

r_l = []

def revise2(code):
    str_list=[]
    code =code.replace("I","1",1)
    st,end =0,0
    for i in range(len(code)-1):
        if code[i] =='[':
            st=i
        elif code[i] ==']':
            end = i            
            break
    if code[end+1] is not '/':
        str_list= list (code)
        str_list.insert(end+1,'/')
        code= "" .join(str_list)
    code=code[:st+1]+re.sub("\D","",code[st:end+1])+code[end:]
    return code

def revise(code):
    code=code.replace("0", "O",3)
    code=code.replace(" ", "",2)
    code=code.replace("|", "l",1)
    code=code.replace("\n", "",5)
    code=code.replace(")", "J",5)
    code=code.replace("£", "Z",1)
    code=code.replace("1/","]/",1)
    code=code.replace("?","S")
    code=code.replace("II","T",1)
    code=code.replace("YD","YDF",1)
    if '[' in code:
        code=code.replace("O", "0",4)
        if code.count('[')==2:
            code=revise2(code)
        if code.count('[') != code.count(']') :
            code=code+"]"
    return code
def idtfy(ws,wb):
    path = "./cut03/"
    for i in range(1,14):
        for j in range(0,11):
            img = cv2.imread(path + str(i)+"_"+str(j)+".jpg")
            img = cv2.copyMakeBorder(img,10,10,10,10, cv2.BORDER_CONSTANT,value=[255,255,255])
            im_h,im_w,tmp=img.shape
            img=cv2.resize(img,(im_w*2,im_h*2))
            code=pytesseract.image_to_string(img,lang='eng',config='-psm 6 ')#digits
            code=revise(code)
            if j==0 :
                r_l.clear()
                r_l.append(code)
            else :
                print(r_l[0]+c_l[j],code)
                ws.append([r_l[0]+c_l[j],code.encode("utf8").decode("cp950", "ignore")])
    # for filename in dirs:
    #     name=filename.split('.', 1 )             
    #     img = cv2.imread(path + filename)
    #     img = cv2.copyMakeBorder(img,10,10,10,10, cv2.BORDER_CONSTANT,value=[255,255,255])
    #     # print(path + filename)
    #     # cv2.imshow("hello",img)
    #     im_h,im_w,tmp=img.shape
    #     img=cv2.resize(img,(im_w*2,im_h*2))
    #     code=pytesseract.image_to_string(img,lang='eng',config='-psm 6 ')#digits
    #     code=revise(code)
    #     print(name[0],code)
    #     # ws.append([name[0],code.encode("utf8").decode("cp950", "ignore")])

wb = Workbook()
ws = wb.active
ws.append(["NUMBER","CONTENT"])
idtfy(ws,wb)
wb.save('FPK_03.xlsx')

