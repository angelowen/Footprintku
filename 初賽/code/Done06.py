import numpy as np
import cv2
import pytesseract,os
import time
from PIL import Image,ImageDraw
import matplotlib.pyplot as plt
from collections import defaultdict
import threading
import multiprocessing as mp
import numpy as np
from openpyxl import Workbook

tStart = time.time()
pytesseract.pytesseract.tesseract_cmd = '.\Tesseract-OCR\\tesseract.exe'
image_color = cv2.imread("./FPK_06.jpg")
im_h ,im_w,color_channel = image_color.shape  

image = cv2.cvtColor(image_color, cv2.COLOR_BGR2GRAY)
adaptive_threshold = cv2.adaptiveThreshold(
    image,
    255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY_INV, 11, 2)

def extract_peek_ranges_from_array(array_vals, minimun_val=200, minimun_range=2):
    start_i = None
    end_i = None
    peek_ranges = []
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
            pass
    return peek_ranges

def Vertical(image, axis=0):
    vertical_sum = np.sum(image, axis=0)
    return vertical_sum
        
def Horizontal(image, axis=1):
    horizontal_sum = np.sum(image, axis=1) 
    return horizontal_sum

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

def cutpic():
    vertical = Vertical(adaptive_threshold/255)
    horizontal= Horizontal(adaptive_threshold/255)

    box_y = np.where(vertical > (im_h/2)) #400大約為圖像高度的一半
    box_x = np.where(horizontal > (im_w/2))
    #消去罪外框線
    # for x in range(0,x-1): 
    for j in box_y:
        adaptive_threshold[:,j] = 0
    for i in box_x:
        adaptive_threshold[i,:]=0   

    vertical_sum = Vertical(adaptive_threshold/255)
    horizontal_sum= Horizontal(adaptive_threshold/255)


    peek_yranges = extract_peek_ranges_from_array(horizontal, minimun_val=5, minimun_range=5)
    peek_ranges = extract_peek_ranges_from_array(vertical, minimun_val=5, minimun_range=9)

    index_y = ['yy','A','B','C','D','E','F','G','H','J','K','L','M','N','P','R','T','U','V','W','Y',
            'AA','AB','AC','AD','yyy']
    index_x = ['xx','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17',
            '18','x']

    path = "./cut06"
    if not os.path.isdir(path):
        os.mkdir(path)

    for i,vertical_range in enumerate(peek_ranges):
        x = vertical_range[0]
        x2 = vertical_range[1]
        for j, horizontal_range in enumerate(peek_yranges):
            y = horizontal_range[0]
            y2 = horizontal_range[1]
            
            
        
            h_sum = np.sum(adaptive_threshold[y:y2,x:x2], axis=1)
            v_sum = np.sum(adaptive_threshold[y:y2,x:x2], axis=0)
            
            adjust_y2 = 0
            adjust_y = 0
            adjust_x2 = 0
            adjust_x = 0
            for p, value in enumerate(h_sum):
                if value > 0:
                    adjust_y = p
                    break
            
            for p, value in enumerate(h_sum[::-1]):
                if value > 0:
                    adjust_y2 = p
                    break
                    
            for p, value in enumerate(v_sum):
                if value > 0:
                    adjust_x = p
                    break
            
            for p, value in enumerate(v_sum[::-1]):
                if value > 0:
                    adjust_x2 = p
                    break
            
            new_y2 = y2 - adjust_y2
            new_y = y + adjust_y
            new_x2 = x2 - adjust_x2
            new_x = x + adjust_x
            

            if (y2-y)>=38 and (x2-x)>=38:
                crop_img = image_color[new_y:new_y2,new_x:new_x2] # y->new_y ...
            else: crop_img = image_color[new_y:new_y2,new_x:new_x2] # y->new_y ...
            im_gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
            bin_im = cv2.adaptiveThreshold(im_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 5, 2)
            cv2.imwrite(path+'\\'+str(index_y[j]) + str(index_x[i]) +'.jpg',bin_im )#儲存

    path = "./cut06"
    path1="./newcut06"
    j = 4 # 2
    i = 16 # 16
    path2 = path +'\\'+str(index_y[j]) + str(index_x[i]) +'.jpg'
    img = cv2.imread(path2)
    w,h,c = img.shape

    plt.imshow(img) 

    img_g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_adaptive_threshold = cv2.adaptiveThreshold(img_g , 255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV, 11, 2)
    plt.imshow(img_g)

    #ver = Vertical(img_adaptive_threshold/255)
    hor = Horizontal(img_adaptive_threshold/255)

    #x_ranges = extract_peek_ranges_from_array(ver, minimun_val=2, minimun_range=3)
    y_ranges = extract_peek_ranges_from_array(hor, minimun_val=4, minimun_range=4)


    if not y_ranges == []:    
        y = y_ranges[0][0]
        y2 = y_ranges[0][1]
        img_g = img_g[y:y2,:]
        img = img[y:y2,:]
    plt.imshow(img_g)
    ret,thresh1 = cv2.threshold(img_g,127,255,cv2.THRESH_BINARY_INV)
    thresh1 = thresh1/255
    black = np.argwhere(thresh1 == 1)
    circle_18=[]
    circle_19=[]
    for k,point in enumerate(black):
        #p1=0
        #p2=0
        a = point[0]
        b = point[1]
        #print(point)
        center1 = [21 ,21]
        center2 = [21 ,22]
        center3 = [22 ,21]
        center4 = [22 ,22]
        d1 = np.linalg.norm(point-center1)
        d2 = np.linalg.norm(point-center2)
        d3 = np.linalg.norm(point-center3)
        d4 = np.linalg.norm(point-center4)
        d = int(1/4*(d1+d2+d3+d4))
        #if d > 19:#或用18
        #    p1 = a
        #    p2 = b
        #    circle_19.append([p1,p2])
        #    img[p1,p2]=255
            
        if d>18:
            p1 = a
            p2 = b
            circle_18.append([p1,p2])
            img[p1,p2]=255
            #print(p1,p2)
    plt.imshow(img) 
    cv2.imwrite(path1+'\\'+ str(index_y[j]) + str(index_x[i]) +'.jpg',img)
    circle_18.remove([8,5])
    circle_18.remove([8,6])



    path1="./newcut06"
    if not os.path.isdir(path1):
        os.mkdir(path1)
    #去圓    
    for j in range(len(peek_yranges)):
        for i in range(len(peek_ranges)): 
            path2 = path +'\\'+str(index_y[j]) + str(index_x[i]) +'.jpg'
            img = cv2.imread(path2)
            w,h,c = img.shape
            plt.imshow(img)             
            if not (w>=44 and h >=44):
                pass
            else:
                for k,p in enumerate(circle_18):
                    p1 = p[0]
                    p2 = p[1]
                    img[p1,p2]=255
                    
                plt.imshow(img)
            #判斷圓內有幾行    
            img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img2_adap = cv2.adaptiveThreshold(
                img2,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV, 11, 2)
            #v = Vertical(img2_adap/255)
            img2_hor = Horizontal(img2_adap/255)
            small_ranges  = extract_peek_ranges_from_array(img2_hor, minimun_val= 2, minimun_range=3)
            #small_ranges = saperate_by_space(img2_hor, space_size=4, minimun_val=5)
            cnt = 0
            for s,small_range in enumerate(small_ranges):
                s1 = small_range[0]
                s2 = small_range[1]
                if  s2-s1 >= 10 and s2-s1 <= 22:
                    img3 = img[s1-3:s2+4,:]
                    cv2.imwrite(path1+'\\'+str(index_y[j]) + str(index_x[i]) +'_'+ str(cnt) +'.jpg' ,img3) 
                    cnt+=1
                elif s2-s1 >= 22:
                    img3 = img[12:32,:]
                    cv2.imwrite(path1+'\\'+str(index_y[j]) + str(index_x[i]) +'_'+ str(cnt) +'.jpg' ,img3)
                #else: 
                #    cv2.imwrite(path1+'\\'+str(index_y[j]) + str(index_x[i]) +'.jpg',img)   
            if small_ranges==[]:
                cv2.imwrite(path1+'\\'+str(index_y[j]) + str(index_x[i]) +'.jpg',img)


def idtfy(ws):
    path = "./newcut06/"
    if not os.path.isdir(path):
        os.mkdir(path)

    past=""
    output=""

    dirs = os.listdir( path )

    for filename in dirs:
        name = [  str(x) for x in filename.split( '.' )]
        name1 = [  str(x) for x in name[0].split( '_' )]
        flag=0
        try:
            if(name1[0][1]=='x'or name1[0][1]=='y' or name1[0][2]=='x'or name1[0][2]=='y'):
                continue
            if(name1[0][-2:]=='10'or name1[0][-2:]=='11' ):
                continue    
        except:
            pass
        for i in range(1,13):
                if (name1[0]=='E'+str(i) or name1[0]=='F'+str(i)or name1[0]=='W'+str(i)or name1[0]=='Y'+str(i)):
                    # print("++")
                    flag=1
                    break
        if(flag==1):
                continue

        img = cv2.imread(path+filename)
        im_h,im_w,tmp=img.shape
        
        
        img = img[:, 2:im_w-2]
        img = cv2.copyMakeBorder(img,0,0,2,2, cv2.BORDER_CONSTANT,value=[255,255,255])
        im_h,im_w,tmp=img.shape
        try:
            a=cv2.resize(img,(im_w*20,im_h*20))
        except :
            a=img
            # print("n")
        try:
            code=pytesseract.image_to_string(a,lang='eng',config='-psm 7 digits')
        except :
            pass
        code=code[:4]
        code=code.replace(" ", "",3)
        code=code.replace("O", "0",3)
        code=code.replace("Z", "2_",3)
        code=code.replace("S", "5",3)
        code=code.replace("[Gas", "GQS",3)
        code=code.replace("ATT", "ATT2",3)
        code=code.replace("TAF", "TA7",3)
        code=code.replace("/", "",3)
        code=code.replace("i*", "A",3)
        code=code.replace("ATT21", "ATT1",3)
        code=code.replace("TAQ", "TA0",3)
        code=code.replace("~", "",3)
        code=code.replace("G05", "GQS",3)
        code=code.replace("GQ5", "GQS",3)
        code=code.replace("_i_", "_",3)


        if len(code)>3 :
            if code[0]=='I':
                code=code[1:]
            if code[1]=='I':
                code=code[0]+'T'+code[2:]

        
                
        last=len(code)-1
        if code!="" and str.isdigit(code[last])==False and str.isalpha(code[last])==False and code[last]!="_": 
            code=code[:last]
        if len(code)==2 and str.islower(code[0])==True and str.islower(code[1])==True:
            code=code[0]+'_'+code[1]
            
        flag=0
        if len(code)==2:
            code=code.replace("AT","ATTA",1)
            if code=="ATTA":
                flag=1
        if flag==0:
                code=code.replace("AL", "AT",3)
        # print(name[0],code,flag)
            

        #######################################
        if(name1[0]==past): 
            output=output+code
            # print(code,output)    
        else:
            last=len(output)-1
            if output!="" and str.isdigit(output[last])==False and str.isalpha(output[last])==False and output[last]!="_": 
                output=output[:last]
            for i in range(len(output)-1):
                try:
                    if len(output)>2 and str.isdigit(output[i])==True and str.isupper(output[i+1])==True:
                        output=output[i]+'_'+output[i+1]
                except :
                    pass 
            print(past,output)
            ws.append([past,output.encode("utf8").decode("cp950", "ignore")])
            output=code
        past=name1[0]


if __name__ == '__main__':
    wb = Workbook()
    ws = wb.active
    ws.append(["NUMBER","CONTENT"])
    cutpic()
    idtfy(ws)
    wb.save('FPK_06.xlsx')
    tEnd = time.time()
    print ("total cost %f sec" % (tEnd - tStart))#會自動做近位
    print (tEnd - tStart)#原型長這樣       