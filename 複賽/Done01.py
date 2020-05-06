import os,sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image,ImageDraw
import pytesseract,os
from collections import defaultdict
import csv
import time
from openpyxl import Workbook
if not os.path.isdir("./01f"):
       os.mkdir("./01f") 
# tStart = time.time()
# print("let's start")
pytesseract.pytesseract.tesseract_cmd = '.\Tesseract-OCR\\tesseract.exe'
image_color = cv2.imread("./FPK_01.jpg")
im_h ,im_w,color_channel = image_color.shape  #　01:(856,1231,3)
image = cv2.cvtColor(image_color, cv2.COLOR_BGR2GRAY)
im = Image.open("./FPK_01.jpg")
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
        else:
            raise ValueError("cannot parse this case...")
    return peek_ranges

def image_sum(image):
    hori =  np.sum(image, axis=1)
    sum_all =  np.sum(hori, axis=0)
    return sum_all/255

vertical = Vertical(adaptive_threshold/255)
horizontal= Horizontal(adaptive_threshold/255)
peek_yranges = extract_peek_ranges_from_array(horizontal, minimun_val=20, minimun_range=5)
peek_xranges = extract_peek_ranges_from_array(vertical, minimun_val=20, minimun_range=5)

vertical_peek_ranges2d = []
for peek_range in peek_yranges:
    start_y = peek_range[0]
    end_y = peek_range[1]
    line_img = adaptive_threshold[start_y:end_y, :]
    vertical_sum = Vertical(line_img)
    vertical_peek_ranges = extract_peek_ranges_from_array(vertical_sum,minimun_val=30,minimun_range=3) 
    vertical_peek_ranges2d.append(vertical_peek_ranges) 
color = (0,0,255)
path="./cut01"
if not os.path.isdir(path):
    os.mkdir(path)
#二值化圖片的空矩陣
cnt = 0
for j, peek_range in enumerate(peek_yranges):
    for i, vertical_range in enumerate(vertical_peek_ranges2d[j]):
        x = vertical_range[0]
        y = peek_range[0]
        x2 = vertical_range[1] 
        y2 = peek_range[1] 
        pt1 = (x, y)
        pt2 = (x2, y2)
        cv2.rectangle(image_color, pt1, pt2, color)
        table = [] 
        cnt +=1
        im2 = im.crop((x,y,x2,y2))
        grey_image = im2.convert('L') #轉為灰度圖
        threshold = np.max(grey_image)*0.65
        for k in range(256):
            if k < threshold:
                table.append(0)
            else:
                table.append(1)       
        bin_image = grey_image.point(table,'1')
        sum_all = image_sum(bin_image)
        #print(str(cnt) , sum_all)
        if sum_all < 10*10:
            bin_image = bin_image.point(lambda x : 1-x)
        
        bin_image.save("./cut01/" +str(cnt) +".jpg")

def add_board(name):
    old_im = Image.open(name)
    old_size = old_im.size
    new_size = (130,130)
    new_im = Image.new("RGB", new_size)   ## luckily, this is already black!
    new_im.paste(old_im, (int((new_size[0]-old_size[0])/2),
                        int((new_size[1]-old_size[1])/2)))
    return new_im

for i in range(1,64):
    name="./cut01/"+str(i)+".jpg"##開啟要辨識的圖片
    with Image.open(name) as img:
        im_h ,im_w = img.size
        if im_w==im_h:
            img = img.rotate(-35)
            img.save('./cut01/r'+str(i)+'.jpg')
        else:
            # img=add_board(name) 
            img.save('./01f/'+str(i)+'.jpg')
            continue     
 
###################################
#圖片需為正方形才可使用函數
def clear_circle(img,i):
    im_h ,im_w,c = img.shape
    radius = int(im_h/2)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret,thresh1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY_INV)
    thresh1 = thresh1/255 
    black = np.argwhere(thresh1 == 1)

    for k,point in enumerate(black):
        a = point[0]
        b = point[1]
        
        center1 =  np.array([155,155])
        center2 =  np.array([155,156])
        center3 =  np.array([156,155])
        center4 =  np.array([156,156])
        #print(np.array([a,b]),center1)
        d1 = np.linalg.norm(point-center1)
        d2 = np.linalg.norm(point-center2)
        d3 = np.linalg.norm(point-center3)
        d4 = np.linalg.norm(point-center4)
        d = int(1/4*(d1+d2+d3+d4))
        
        if d > radius-22 :
            p1 = a
            p2 = b
            img[p1,p2] = 255
            #print(p1,p2)
        
    plt.imshow(img) 
    cv2.imwrite('./01f/'+str(i)+'.jpg',img)
    
   

for i in range(1,64):
    try:
        name= "./cut01/r"+str(i)+".jpg" ##開啟要辨識的圖片
        img = cv2.imread(name)
        im_h ,im_w ,c= img.shape
        if im_w == im_h:
            clear_circle(img,i)
    except :
        pass

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
for i in range(1,64):
    name='./01f/'+str(i)+'.jpg'
    img = cv2.imread(name) 
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

wb.save('FPK_01.xlsx')

