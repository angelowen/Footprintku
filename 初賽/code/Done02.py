
import numpy as np
import cv2,csv
import pytesseract,os
import time
from PIL import Image,ImageDraw ,ImageOps
import matplotlib.pyplot as plt
from collections import defaultdict
from openpyxl import Workbook
#缺坐標辨識

# %load cut02bin-box.py
#!/usr/bin/env python
import os

tStart = time.time()
pytesseract.pytesseract.tesseract_cmd = '.\Tesseract-OCR\\tesseract.exe'


print("Let's start")

name  = "./cut02bin/"
if not os.path.isdir(name):
    os.mkdir(name)

# 行或列加總後，印出長條圖
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

def extract_peek_ranges(array_vals, minimun_val, minimun_range):
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

def img_crop(img,lis):
    [y1,y2,x1,x2]=lis
    img_crop = img[y1:y2,x1:x2]
    #crop_resize = cv2.resize(img_crop,((x2-x1)*2,(y2-y1)*2))
    return img_crop

def box_cut(im,left, upper, right, lower):
    region = im.crop((left, upper, right, lower))
    return region




image_color = cv2.imread("./FPK_02.jpg")
im_h ,im_w,color_channel = image_color.shape 
img = cv2.cvtColor(image_color, cv2.COLOR_BGR2GRAY)
adaptive_threshold = cv2.adaptiveThreshold(
    img,
    255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY_INV, 11, 2)
im = Image.open("./FPK_02.jpg")

index_x=[]
for i in range(1,36):
    index_x.append(str(i))
index_y = ['A','B','C','D','E','F','G','H','J','K','L','M','N','P','R','T','U','V','W',
           'Y','AA','AB','AC','AD','AE','AF','AG','AH','AJ','AK','AL','AM','AN','AP','AR','AT',
           'AU','AV','AW','AY','BA','BB','BC','BD','BE','BF','BG','BH','BJ','BK']

vertical = Vertical(adaptive_threshold/255)
horizontal= Horizontal(adaptive_threshold/255)

box_x = np.where(vertical > (im_h/3)) #400大約為圖像高度的一半
box_y = np.where(horizontal > (im_w/3))

Box_x = box_x[0]
listx = Box_x.tolist()
Box_y = box_y[0]
listy = Box_y.tolist()
#print(Box_x,Box_y)
X=[]
Y=[]
for i in range(len(Box_x)):
    if i%3==1:
        X.append(Box_x[i])
        
for j in range(len(Box_y)):
    if j%3==1:
        Y.append(Box_y[j])      
        
        

cnt = 0
table = []

wb = Workbook()
ws = wb.active
ws.append(["NUMBER","CONTENT"])

for j in range(len(Y)-1):
    for i in range(len(X)-1):
        table = []
        new = box_cut(im,X[i]+4,Y[j]+4,X[i+1]-2,Y[j+1]-2)
        grey_image = new.convert('L')
        #grey_image.save("./cut02g/" +str(cnt) +".jpg")

        width = X[i+1]-X[i]-6
        height = Y[j+1]-Y[j]-6

        threshold = np.max(grey_image)*0.9
        for k in range(256):
            if k < threshold:
                table.append(0)
            else:
                table.append(1)

        bim = grey_image.point(table, '1')
        s = sum(Horizontal(bim)/255)
        # print(s)
        # if  s.all()<= 5:
        #     print("pass")
        #     continue
        bim = bim.resize((width*2, height*2))
        bim = ImageOps.expand(bim, border=(10,0,10,10), fill=255)##left,top,right,bottom
        code=pytesseract.image_to_string(bim,lang='eng',config='--psm 6')#,lang='eng',config='-psm 7 digits'

        ##
        code=code.replace("\n", "", 3)
        code=code.replace("(4", "_", 1)
        code=code.replace("(U", "_", 1)
        code=code.replace("(", "", 1)
        code=code.replace(" ", "_", 3)
        code=code.replace("vV", "V", 1)
        code=code.replace("v", "V", 1)
        code=code.replace("xX", "X", 1)
        code=code.replace("x", "X", 1)
        code=code.replace("¥", "", 1)
        code=code.replace("g", "", 1)
        ode=code.replace("o", "0", 1)
        if len(code)>=4 and (code[:4]=="EBKO" or code[:4]=="TLKO" or code[:4]=="DLKO"):
            code=code[:3]+'0'+code[4:]
        if len(code)>=5 and (code[:5]=="ATKEO"):
            code=code[:4]+'0'+code[5:]
        if (len(code)>0 and str.islower(code[0])==True):
            code=str.upper(code[0])+code[1:]
        if (len(code)>0 and str.isalpha(code[0])==False):
            code=code[1:]

        #name = [str(x) for x in filename.split( '.' )]
        # print(index_y[j] + index_x[i],code)
        ws.append([index_y[j] + index_x[i],code.encode("utf8").decode("cp950", "ignore")])
        ##

        if( time.time()-tStart > 130):
            wb.save('FPK_02.xlsx')
            tEnd = time.time()
            print ("It cost %f sec" % (tEnd - tStart))#會自動做近位
            print (tEnd - tStart)#原型長這樣
            exit(0)
        
        #print(index_y[j] + index_x[i],code)
        #bim.save("./cut02bin/" +index_y[j] + index_x[i] +".jpg")



