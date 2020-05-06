#!/usr/bin/env python
# coding: utf-8
import numpy as np
import pytesseract
import os
import cv2
import matplotlib.pyplot as plt
from PIL import Image,ImageDraw
from collections import defaultdict
import time
from openpyxl import Workbook

tStart = time.time()
pytesseract.pytesseract.tesseract_cmd = '.\Tesseract-OCR\\tesseract.exe'

print("Start!! it will not show the output\n")
im = Image.open("./FPK_04.jpg")
path = "./cut04/"
if os.path.isdir(path):
    command = "rmdir /s /q \"%s\""
    command = command % path
    result = os.system(command)
os.mkdir(path)

index_x=[]
for i in range(0,23+2):
    index_x.append(str(i))

index_y = ['index1','A','B','C','D','E','F','G','H','J','K','L','M','N','P','R','T','U','V','W',
           'Y','AA','AB','AC','index2']

def box_cut(left, upper, right, lower):
    region = im.crop((left, upper, right, lower))
    return region

def box_ranges_from_array(lis,a,b):

    if lis[b]-lis[a] == 1:
        return False
    else:
        return True
def cutpic():
    image_color = cv2.imread("./FPK_04.jpg")
    im_h ,im_w,color_channel = image_color.shape 
    image = cv2.cvtColor(image_color, cv2.COLOR_BGR2GRAY)
    adaptive_threshold = cv2.adaptiveThreshold(
        image,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2)
    vertical_sum = np.sum(adaptive_threshold, axis=0)
    vertical = vertical_sum/255

    horizontal_sum = np.sum(adaptive_threshold, axis=1)
    horizontal= horizontal_sum/255
    #使用plt.plot畫(x ,y)曲線

    box_x = np.where(vertical > (im_h/2)) #400大約為圖像高度的一半
    box_y = np.where(horizontal > (im_w/2))

    Box_x = box_x[0]
    listx = Box_x.tolist()
    #print(Box_x)

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

    index_x=[]
    for i in range(0,23+2):
        index_x.append(str(i))
    index_y = ['index1','A','B','C','D','E','F','G','H','J','K','L','M','N','P','R','T','U','V','W',
            'Y','AA','AB','AC','index2']



    blank_table = [] #二值化圖片的空矩陣
    for j in range(int(len(listy))-1):
        if box_ranges_from_array(listy,j,j+1): 
            for i in range(int(len(listx))-1):
                if box_ranges_from_array(listx,i,i+1):
                    table = []
                    new = box_cut(listx[j],listy[i],listx[j+1],listy[i+1]) #裁成小圖
                    grey_image = new.convert('L') #轉為灰度圖
                    grey_image.save("./cut04/" +str(index_y[i])+str(index_x[j]) +".jpg")
                    
                    #二值化圖片#有些效果不好
                    threshold = np.max(grey_image)*0.85
                    for k in range(256):
                        if k < threshold:
                            table.append(0)
                        else:
                            table.append(1)
                    ##table = find_array_frame(256,threshold,blank_table)
                    
                    bin_image = grey_image.point(table, '1')
                    #bin_image.save("C:/Users/USER/Desktop/Game/cut04/" +str(cnt) +".jpg")   

    
def identify(ws):
    flag1=0
    plus_cnt=0
    dirs = os.listdir( path )

    for filename in dirs:
        blank=0
        name = [  str(x) for x in filename.split( '.' )]
        for k in range(3,22):
            if name[0]=='C'+str(k) or name[0]=="G"+str(k) or name[0]=='K'+str(k) or name[0]=='P'+str(k) or name[0]=="U"+str(k) or name[0]=="AA"+str(k):
                blank=1
                break
        for k in range(3,22):
            if name[0]==index_y[k]+'3' or name[0]==index_y[k]+'7'or name[0]==index_y[k]+'10'or name[0]==index_y[k]+'14'or name[0]==index_y[k]+'17'or name[0]==index_y[k]+'21':
                blank=1
                break
        if blank==1:
            continue
        
        flag=0
        img = cv2.imread(path+filename)
        im_h,im_w,tmp=img.shape
        img = img[5:im_h-5, 2:im_w-2]
        # img = cv2.copyMakeBorder(img,5,5,5,5, cv2.BORDER_CONSTANT,value=[255,255,255])
        im_h,im_w,tmp=img.shape
        # print(a.shape)
        a=cv2.resize(img,(im_w*10,im_h*10))
        # cv2.imshow("img",a)
        # cv2.waitKey(0)
        code=pytesseract.image_to_string(a,lang='eng',config='-psm 6 digits')#,lang='eng',config='-psm 7 digits'
        flag=0
        for k in range(len(code)):
            if(code[k]=='_'):
                flag=1
        if flag==0:
            code=code.replace(" ", "_",2)
        code=code.replace(" ", "",5)
        code=code.replace("\'", "",5)
        code=code.replace("\n", "",5)
        code=code.replace("O", "0",3)
        code=code.replace("o", "0",3)
        code=code.replace("I1", "1",1)
        code=code.replace("Xx", "XX",1)
        code=code.replace("XxX", "XX",1)
        code=code.replace("xXx", "XX",1)
        code=code.replace("XXx", "XX",1)
        code=code.replace("0_P", "0P",1)
        if len(code)>5 and code[:3]=='WLL' and code[3]!='_':
                code=code[:3]+'_'+code[3:]
        if len(code)>3 and str.isupper(code[len(code)-1])==False:
                code=code[:len(code)-1]
        if len(code)>=8 :
            for k in range(len(code)):
                if str.isdigit(code[k])==True and k+1<len(code) and code[k+1]!='_':
                    code=code[:k+1]+"_"+code[k+1:]
        if len(code)>1 and code[len(code)-2]=='X':
            if int(plus_cnt/2)%2==0:
                code+='1'
            else:
                code+='2'
            plus_cnt+=1


        if code.find("_LWR_PWB_WMW")!=-1:
            code="WXX0P9_LWR_PWB_WMW"
        if len(code)==12 and code[8]=='I':
            code=code[:8]+'1'+code[9:]
        
        print(name[0],code)
        ws.append([name[0],code.encode("utf8").decode("cp950", "ignore")])

if __name__ == '__main__':
    wb = Workbook()
    ws = wb.active
    ws.append(["NUMBER","CONTENT"])
    cutpic()
    identify(ws)
    wb.save('FPK_04.xlsx')
    tEnd = time.time()
    print ("total cost %f sec" % (tEnd - tStart))#會自動做近位
    print (tEnd - tStart)#原型長這樣