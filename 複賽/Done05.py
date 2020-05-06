
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
pytesseract.pytesseract.tesseract_cmd = '.\Tesseract-OCR\\tesseract.exe'
image_color = cv2.imread("./FPK_05.jpg")
im_h ,im_w,color_channel = image_color.shape 
image = cv2.cvtColor(image_color, cv2.COLOR_BGR2GRAY)
adaptive_threshold = cv2.adaptiveThreshold(
    image,
    255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY_INV, 11, 2)
im = Image.open("./FPK_05.jpg")


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

vertical = Vertical(adaptive_threshold/255)
horizontal= Horizontal(adaptive_threshold/255)
box_x = np.where(vertical > (im_h/2)) #400大約為圖像高度的一半
box_y = np.where(horizontal > (im_w/2))
print(np.where(horizontal > (im_w/2)))
      
#去掉大框
for i in box_y:
    adaptive_threshold[i,:] = 0
for j in box_x:
    adaptive_threshold[:,j]=0
    
horizontal= Horizontal(adaptive_threshold/255)
#使用plt.plot畫(x ,y)曲線

vertical = Vertical(adaptive_threshold/255)

Box_x = box_x[0]
listx = Box_x.tolist()

Box_y = box_y[0]
listy = Box_y.tolist()



# In[49]:
#A = extract_peek_ranges_from_array(vertical,2,70)
#B = extract_peek_ranges_from_array(horizontal,10,22)


# In[3]:


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

A = saperate_by_space(vertical,4,10)
B = saperate_by_space(horizontal,4,10)
print(A)
print(B)

A = [x for x in A if x[1] - x[0] > 100]
B = [x for x in B if x[1] - x[0] > 100]
path=".\cut05"
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
        new = box_cut(x1,y1,x2,y2) #裁成小圖
        grey_image = new.convert('L') #轉為灰度圖
        threshold = 200
        for k in range(256):
            if k < threshold:
                  table.append(0)
            else:
                table.append(1)
        ##table = find_array_frame(256,threshold,blank_table)
                
        bin_image = grey_image.point(table, '1')
        bin_image.save("./cut05/" +str(cnt) +".jpg")   
        cnt +=1        

def clear_circle(img,i):
    im_h ,im_w,c = img.shape
    radius = 75
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret,thresh1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY_INV)
    thresh1 = thresh1/255 
    black = np.argwhere(thresh1 == 1)
    for k,point in enumerate(black):
        a = point[0]
        b = point[1]
        
        center1 =  np.array([75,75])
        center2 =  np.array([75,76])
        center3 =  np.array([76,75])
        center4 =  np.array([76,76])
        #print(np.array([a,b]),center1)
        d1 = np.linalg.norm(point-center1)
        d2 = np.linalg.norm(point-center2)
        d3 = np.linalg.norm(point-center3)
        d4 = np.linalg.norm(point-center4)
        d = int(1/4*(d1+d2+d3+d4))

        
        if d > radius-28 :
            p1 = a
            p2 = b
            #circle.append([p1,p2])
            #print(img[p1,p2])
            img[p1,p2] = 255
            #print(p1,p2)
        
    plt.imshow(img)
    im_h,im_w=img.shape
    img = img[45:im_h-55, 30:im_w-35]
    img= cv2.copyMakeBorder(img,2,2,2,2, cv2.BORDER_CONSTANT,value=[255,255,255])
    cv2.imwrite('./cut05/'+str(i)+'.jpg',img)
    
    # circle_18.remove([8,5])
    # circle_18.remove([8,6])

cnt = 0    
for j in range(len(B)):
    for i in range(len(A)):
        name= "./cut05/"+str(cnt)+".jpg" ##開啟要辨識的圖片
        img = cv2.imread(name)
        im_h ,im_w ,c= img.shape
        u = int((150-im_h)/2)
        v = int((150-im_w)/2)
        img = cv2.copyMakeBorder(img,u,150-im_h-u,v,150-im_w-v, cv2.BORDER_CONSTANT,value=[255,255,255])       
        clear_circle(img,cnt)
        cnt +=1 

list_c=[str(i) for i in range(1,10)]
list_c.pop(3)
list_c.pop(3)
list_c.pop(3)

list_r=[chr(ord('A')+i) for i in range(11)]
list_r.pop(8)
ans=[]

def revise(code):
    code=code.replace("O", "0",3)
    code=code.replace(" ", "",2)
    code=code.replace("T", "1",1)
    code=code.replace("B", "8",1)
    code=code.replace("HJ", "HU",1)
    code=code.replace("NZ", "N2",1)
    code=code.replace("-", "",1)
    code=code.replace("_", "",1)
    code=code.replace("I", "\\",1)
    code=code.replace("/", "7",1)
    code=code.replace("M", "N",1)
    if len(code)<=4:
            code=code.replace("S", "5",1)
    if len(code)>1 :
        if code[0]=='0':
            code='O'+code[1:]
        elif code[0]=='2':
            code='Z'+code[1:]
        elif code[0]=='5':
            code='S'+code[1:]
        elif code[0]=='1':
            code='T'+code[1:]
    if len(code)>=3 :
        if code[2]=='G':
            code=code[:-1]+'6'
        elif code[2]=='Z':
            code=code[:-1]+'7'
    if ("R" in code) and (code[0] is not'R'):
        code="ZKHQ"
    return code

def idtfy(ws,wb):
    path = "./cut05/"
    if not os.path.isdir(path):
        os.mkdir(path)
    dirs = os.listdir( path )
    for i in range(0,60):
        name=path + str(i)+".jpg"           
        img = cv2.imread(name)
        im_h,im_w,tmp=img.shape   
        img=cv2.resize(img,(im_w*10,im_h*10))
        # cv2.imshow("hello",img )
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        code=pytesseract.image_to_string(img,lang='eng',config='-psm 6 digits')
        code=revise(code)
        print(code)
        ans.append(code)

    cnt=0
    for i in list_r:
            for j in list_c:
                print(i+j,ans[cnt])
                ws.append([i+j,ans[cnt].encode("utf8").decode("cp950", "ignore")])
                cnt+=1

wb = Workbook()
ws = wb.active
ws.append(["NUMBER","CONTENT"])
idtfy(ws,wb)
wb.save('FPK_05.xlsx')