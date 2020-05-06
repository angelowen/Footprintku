
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

tStart = time.time()
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

#im是使用PIL開啟圖
def box_cut(im,left, upper, right, lower):
    #img_name = "./FPK_04.jpg"
    #im = Image.open(img_name)
    region = im.crop((left, upper, right, lower))
    return region

#檢查表格位置是否正確，可調==1處
def box_ranges_from_array(lis,a,b):
    if lis[b]-lis[a] == 1:
        return False
    else:
        return True

def image_sum(image):
    hori =  np.sum(image, axis=1)
    sum_all =  np.sum(hori, axis=0)
    return sum_all/255
    
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
        #else:
        #    raise ValueError("cannot parse this case...")
    return peek_ranges


#TorF:是否要另外切座標,fra:圖像高度或寬度的多少
def get_cut_coor(TorF,adaptive_threshold,fra,im_w ,im_h):   
    vertical = Vertical(adaptive_threshold/255)
    horizontal= Horizontal(adaptive_threshold/255)
    
    box_x = np.where(vertical > (im_h*fra)) #大約為圖像高度的??
    box_y = np.where(horizontal > (im_w*fra))
    Box_x = box_x[0]
    listx = Box_x.tolist()
    #print(Box_x)
    Box_y = box_y[0]
    listy = Box_y.tolist()
    
    #可調整參數決定範圍(框線厚度)
    ran = 4
    for k1 in range(len(listx)-1):
        if listx[k1+1]-listx[k1] < ran:
            listx[k1]=0

    for k2 in range(len(listy)-1):    
        if listy[k2+1]-listy[k2] < ran:
            listy[k2]=0

    while 0 in listx:        
        listx.remove(0)
    while 0 in listy:
        listy.remove(0)
    
    #需調整座標裁切的space_size.minimun_val
    if TorF:
        coor1 = saperate_by_space(vertical , space_size=5, minimun_val=3)
        coor2 = saperate_by_space(horizontal , space_size=5, minimun_val=10)
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
        #將座標的位置起始，插入陣列第一個
        listx.insert(0,sx) 
        listy.insert(0,sy) 
        
        return listx,listy

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
    #print(str(cnt) , sum_all)
    if sum_all < sum_value and i > 0 and j > 0:
        bin_image = bin_image.point(lambda x : 1-x)
        
    return bin_image

#去除多餘框線，可調參數(大於圖片長或寬的多少)
def clear_line(img_name):
    image = cv2.imread(img_name,0)
    im_h ,im_w = image.shape
    ret,adaptive = cv2.threshold(
        image,
        190,
        255,
        cv2.THRESH_BINARY_INV)

    #找出多於框限
    vertical = Vertical(adaptive)/255
    horizontal = Horizontal(adaptive)/255
    small_box_x = np.where(vertical > (3*im_h/4)) #400大約為圖像高度的一半
    small_box_y = np.where(horizontal > (3*im_w/4))

    #去掉框線
    for k in small_box_y:
        adaptive[k,:] = 0
    for k in small_box_x:
        adaptive[:,k]=0    

    #原圖去框
    for k in small_box_y:
        image[k,:]=255
    for k in small_box_x:
        image[:,k]=255
    
    return image

#用3*3的中值滤波器
def m_filter(x,y):
    sum_s=[]
    for k in range(-int(step/2),int(step/2)+1):
        for m in  range(-int(step/2),int(step/2)+1):
            e = im[x+k][y+m]
            sum_s.append(e)
    sum_s.sort()
    return sum_s[(int(step*step/2)+1)]

def image_name(file_name,index_x,index_y,i,j):
    final = file_name+"/"+str(index_y[j]) + str(index_x[i])+".jpg"
    return final

##結束函式定義


# In[9]:


#圖片檔案名稱
img_name = "./FPK_08.jpg"
im = Image.open(img_name)
pytesseract.pytesseract.tesseract_cmd = './Tesseract-OCR/tesseract.exe'
adaptive_threshold,im_w ,im_h = cv2_to_adaptive__threshold(img_name)

#圖片儲存的資料夾
name = "./cut08"
if not os.path.isdir(name):
    os.mkdir(name)
    
##儲存其他圖片的資料夾
#name2 = "./cut08old"
#if not os.path.isdir(name2):
#    os.mkdir(name2)
##
    
listx,listy = get_cut_coor(True,adaptive_threshold,7/18,im_w ,im_h)#需調參數，獲得表格座標
step=3
index_y = ["0","C","E","G","H","J","K","L","M","P"]
index_x = [str(i) for i in range(0,10)]
for j in range(int(len(listy))-1):
    if box_ranges_from_array(listy,j,j+1): 
        for i in range(int(len(listx))-1):
            if box_ranges_from_array(listx,i,i+1):
                
                if i==0 or j==0:
                    new = box_cut(im,listx[i]-3,listy[j]-1,listx[i+1]-3,listy[j+1]-3)
                else:
                    new = box_cut(im,listx[i]+5,listy[j]+5,listx[i+1]-5,listy[j+1]-5)
                bin_image = color_image_threshold(new ,0.85,i,j,25)
                
                #照片儲存與讀取名稱
                img_name = image_name(name,index_x,index_y,i,j)#可存name2
                bin_image.save(img_name)#存完二值化小圖
                
                #去除多餘框線，可調函數內參數(大於圖片長或寬的多少)
                image = clear_line(img_name)
                cv2.imwrite(img_name,image)
                #存去邊框小圖
                
                
                #條件去噪
                #if (j==3and (i==1or i ==9))or(j==5and i==2)or(j==7and i==3):
                """
                im_copy=[]
                im=io.imread(img_name)
                im_copy = im.copy()
                for ii in range(int(step/2),im.shape[0]-int(step/2)):
                    for jj in range(int(step/2),im.shape[1]-int(step/2)):
                        im_copy[ii][jj]= m_filter(ii,jj)
                io.imsave(img_name,im_copy)
                """
                    # 上面or下面 #兩種方法#
                if (j==1and i==5)or(j==3 and (i==1or i ==6or i==9))or(j==4and i==2)or(j==5and (i==2or i==8))or(j==6and (i==3or i==8))or(j==7and (i==3or i==5))or(j==9and (i==2 or i==6)):    
                    img = cv2.imread(img_name)
                    img = cv2.medianBlur(img,3)
                    ret,img = cv2.threshold(img,175,255,cv2.THRESH_BINARY) 
                    cv2.imwrite(img_name,img)

                
tEnd = time.time()
print("cost time :",tEnd-tStart)
################################################################


pytesseract.pytesseract.tesseract_cmd = '.\Tesseract-OCR\\tesseract.exe'
def revise2(code):
    st,end =0,0
    for i in range(len(code)-1):
        if code[i] =='[':
            st=i
        elif code[i] ==']':
            end = i            
            break
    str_list=[]
    if code[end+1] is not '/':
        str_list= list (code)
        # nPos=str_list.index( '/' ) ##可略
        str_list.insert(end+1,'/')
        code= "" .join(str_list)
    code=code[:st+1]+re.sub("\D","",code[st:end+1])+code[end:] ##re.sub->刪除非字母
    return code

def revise(code):
    code=code.replace("O", "0",3)
    code=code.replace(" ", "",5)
    code=code.replace("\n", "",5)
    code=code.replace("BK","B/K")
    code=code.replace("VIM","V/M")
    code=code.replace("s","8")
    code=code.replace("__","_")
    code=code.replace("0IN","01N")
    code=code.replace("P0","P_0")
    code=code.replace("o","0")
    code=code.replace("00","0")
    code=code.replace("0G","09")
    code=code.replace("0S","05")
    code=code.replace("DA","D")
    code=code.replace("vV","V")
    if len(code)>1 :
        if code[0]=='0':
            code='O'+code[1:]
        if code[0]=='1':
            code='I'+code[1:]
    if code[-2:]=="NT":
        code="QMOINT"       
    return code

def idtfy(ws,wb):
    path = "./cut08/"###改
    dirs = os.listdir( path )
    for filename in dirs:
        name=filename.split('.', 1 )             
        img = cv2.imread(path + filename)
        # img = cv2.copyMakeBorder(img,10,10,10,10, cv2.BORDER_CONSTANT,value=[255,255,255])#### 加白框
        im_h,im_w,tmp=img.shape
        img=cv2.resize(img,(im_w*10,im_h*10))
        code=pytesseract.image_to_string(img,lang='eng',config='-psm 6')# ,  if block用6
        code=revise(code)
        print(name[0],code)
        ws.append([name[0],code.encode("utf8").decode("cp950", "ignore")])


def write_xls():
    wb = Workbook()
    ws = wb.active
    ws.append(["NUMBER","CONTENT"])
    idtfy(ws,wb)
    wb.save('Team_025_08.xlsx')###改




write_xls()




