
import numpy as np
import cv2,csv
import os
import matplotlib.pyplot as plt
import math
import pytesseract
from PIL import Image,ImageDraw
import pytesseract,os
from collections import defaultdict
import csv
import time
from openpyxl import Workbook

# In[2]:


# range from array
def extract_peek_ranges_from_array(array_vals, minimun_val, minimun_range):
    start_i = None
    end_i = None
    peek_ranges = []
    #enumerate() 函數用於將數據對象組合為一個索引序列，同時列出數據和數據下標
    for i, val in enumerate(array_vals):
        if val >= minimun_val and start_i is None:
            start_i = i
        elif val >= minimun_val and start_i is not None:
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



def saperate_by_space(array_vals, minimun_val, space_size):    
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


# In[3]:


#function for cutting picture


def ranged_horizontal(image, range_tuple = None, minimun_val = 1, minimun_range = 2):
    if range_tuple == None:
        range_tuple = rangetuple_of_entire_image(image)
        
    horizontal_sum = np.sum(spot_image(image, range_tuple), axis=1)
    #plt.plot(horizontal_sum)
    #plt.show
    UDranges = extract_peek_ranges_from_array(horizontal_sum, minimun_val, minimun_range)
    start_point = range_tuple[0][0]; 
    UDranges_in_image = [(start_point + U, start_point + D) for U, D in UDranges]
    list_of_range_tuples = [(UDrange_tuple, range_tuple[1]) for UDrange_tuple in UDranges_in_image]
    return list_of_range_tuples



def ranged_vertical(image, range_tuple = None, minimun_val = 1, minimun_range = 2):
    if range_tuple == None:
        range_tuple = rangetuple_of_entire_image(image)
        
    vertical_sum = np.sum(spot_image(image, range_tuple), axis=0)
    LRranges = extract_peek_ranges_from_array(vertical_sum, minimun_val, minimun_range)
    start_point = range_tuple[1][0]; 
    LRranges_in_image = [(start_point + L, start_point + R) for L, R in LRranges]
    list_of_range_tuples = [(range_tuple[0], LRrange_tuple) for LRrange_tuple in LRranges_in_image]
    return list_of_range_tuples



def ranged_horizontal_by_space(image, range_tuple = None, minimun_val = 1, space_len = 1):
    if range_tuple == None:
        range_tuple = rangetuple_of_entire_image(image)
        
    horizontal_sum = np.sum(spot_image(image, range_tuple), axis=1)
    UDranges = saperate_by_space(horizontal_sum,minimun_val,space_len)
    start_point = range_tuple[0][0]; 
    UDranges_in_image = [(start_point + U, start_point + D) for U, D in UDranges]
    list_of_range_tuples = [(UDrange_tuple, range_tuple[1]) for UDrange_tuple in UDranges_in_image]
    return list_of_range_tuples



def ranged_vertical_by_space(image, range_tuple = None, minimun_val = 1, space_len = 1):
    if range_tuple == None:
        range_tuple = rangetuple_of_entire_image(image)
        
    horizontal_sum = np.sum(spot_image(image, range_tuple), axis=0)
    LRranges = saperate_by_space(horizontal_sum,minimun_val,space_len)
    start_point = range_tuple[1][0]; 
    LRranges_in_image = [(start_point + L, start_point + R) for L, R in LRranges]
    list_of_range_tuples = [(range_tuple[0], LRrange_tuple) for LRrange_tuple in LRranges_in_image]
    return list_of_range_tuples


# In[4]:


# function for range-tuple
# form of range-tuple: ((top, bottom), (left, right))

def rangetuple_of_entire_image(np2Dimage):
    vertical, horizontial = np2Dimage.shape
    top = 0
    bottom = vertical-1
    left = 0
    right = horizontial -1
    range_tuple = ((top, bottom), (left, right))
    return range_tuple

# 可以輸出指定範圍的圖片
def spot_image(image, range_tuple):
    ((top, bottom), (left, right)) = range_tuple
    new_image = image[top : bottom, left : right]
    return new_image


# In[5]:


# processing for table cutting

def clear_muti_lable(peek, label, axis):
    size = label[axis][1] - label[axis][0]
    for i, val in enumerate(peek):
        if val[axis][1] - val[axis][0] < 1.5 * size :
            peek.pop(i)
            
def index_of_min_range(peek_ranges, axis):
    i_min = None
    range_min = None
    for i, peek_range in enumerate(peek_ranges):
        if range_min is None or peek_range[axis][1] - peek_range[axis][0] < range_min :
            range_min = peek_range[axis][1] - peek_range[axis][0]
            i_min = i
    return i_min
 
    
def clear_muti_lable(peek, label, axis):
    size = label[axis][1] - label[axis][0]
    for i, val in enumerate(peek):
        if val[axis][1] - val[axis][0] < 1.5 * size :
            peek.pop(i)


# In[6]:


# picture processing

# clean the circle
# one of these can clean the box

# > 10011 <
# > 10011 <
# > 10001 <
# > 10001 <
def clean_left_right_bounding(polar_image, pixel_threshold):
    for row in polar_image:
        cleaner = False;
        for i,pixel in enumerate(row):
            if cleaner == True and pixel < pixel_threshold :
                cleaner = False;
                break;
            if pixel >= pixel_threshold :
                cleaner = True;
            if cleaner == True :
                row[i] = 0;

        for i,pixel in reversed(list(enumerate(row))):
            if cleaner == True and pixel < pixel_threshold :
                cleaner = False;
                break;
            if pixel >= pixel_threshold :
                cleaner = True;
            if cleaner == True :
                row[i] = 0;
    return



#  vvvvv
#  11111 
#  10011 
#  00001 
#  11111
#  ^^^^^
def clean_top_bottom_bounding(polar_image, pixel_threshold):
    bottom, right = polar_image.shape
    for column in range(right-1):
        cleaner = False;
        for row in range(bottom-1):
            pixel = polar_image[row][column]
            if cleaner == True and pixel < pixel_threshold :
                cleaner = False;
                break;
            if pixel >= pixel_threshold :
                cleaner = True;
            if cleaner == True :
                polar_image[row][column] = 0;

        for row in reversed(range(bottom-1)):
            pixel = polar_image[row][column]
            if cleaner == True and pixel < pixel_threshold :
                cleaner = False;
                break;
            if pixel >= pixel_threshold :
                cleaner = True;
            if cleaner == True :
                polar_image[row][column] = 0;
    return



# In[7]:


def cvshowimage(npimage):
    cv2.imshow('image',npimage)
    cv2.waitKey(0),
    cv2.destroyAllWindows()


# In[8]:


def picture_input(file_address):
    image_color = cv2.imread(file_address)
    image = cv2.cvtColor(image_color, cv2.COLOR_BGR2GRAY)
    thresholded_image = cv2.adaptiveThreshold(
        image,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2)
    # make image input success
    thresholded_image.shape
    return thresholded_image


# In[ ]:





# In[9]:


file_address = "./FPK_06.jpg"
thresholded_image = picture_input(file_address)


# In[10]:


def saperate_label_content(thresholded_image):
    horizontal_cut = ranged_horizontal(thresholded_image)
    column_label = horizontal_cut.pop(index_of_min_range(horizontal_cut, 0))
    clear_muti_lable(horizontal_cut, column_label, 0)

    vertical_cut = ranged_vertical(thresholded_image, horizontal_cut[0])
    row_label = vertical_cut.pop(index_of_min_range(vertical_cut, 1))
    clear_muti_lable(vertical_cut, row_label, 1)

    row_labels_list = ranged_horizontal(thresholded_image, row_label)
    column_labels_list = ranged_vertical_by_space(thresholded_image, column_label,20, 10)

    contents_list = vertical_cut
    
    return column_labels_list, row_labels_list, contents_list


# In[11]:


column_labels_list, row_labels_list, content_list =  saperate_label_content(thresholded_image)


# In[12]:


column_labels_list.pop(0)


# In[13]:


#adjust
def contents_processing(thresholded_image, content_list):
    image = thresholded_image.copy()
    clean_left_right_bounding(spot_image(image, content_list[0]), 100)
    return image


# In[14]:


cutable_content_pic = contents_processing(thresholded_image ,content_list)


# 
# horizontal_bars = ranged_horizontal_by_space(polar_image, content_list[0], 2)
# row_bar_with_label_dic = match_label_row_bar(list_row_label)
# 
# # or
# column_bar_with_label_dic = match_label_column_bar(list_column_label)

# In[15]:


horizontal_bars = ranged_horizontal_by_space(cutable_content_pic, content_list[0], 1, 10)


# In[16]:


def match_label_row_bar(row_labels_list, horizontal_bars, adjust):
    di = {}
    for i, label in enumerate(row_labels_list):
        mid = (label[0][0]+label[0][1])/2 + adjust
        for bar in horizontal_bars:
            (top, bottom),(_,_) = bar
            if(top < mid < bottom):
                di[i] = bar
    row_bar_with_label = di
    return row_bar_with_label


# In[17]:


row_bar_with_label = match_label_row_bar(row_labels_list, horizontal_bars, 0)


# In[18]:


row_bar_with_label[0] = ((145 + 26 , 295), (103, 2127))


# In[19]:


def all_contents_from_rowbar(cutable_content_pic, row_bar_with_label, list_columns_label):
    all_contents = {}
    for row in row_bar_with_label:
        ## ranged_vertical_by_space(image, range_tuple = None, minimun_val = 1, space_len = 1)
        littlepics = ranged_vertical_by_space(cutable_content_pic, row_bar_with_label[row], 10, 10)
        di = {}
        for strpic in littlepics:
            left = strpic[1][0]
            right = strpic[1][1]
            for i_column, column_label in enumerate(list_columns_label) :
                label_mid = (column_label[1][0] + column_label[1][1])/2
                if  left < label_mid < right :
                    di[i_column] = strpic

        all_contents[row] = di
    
    return all_contents


# In[20]:


all_contents = all_contents_from_rowbar(cutable_content_pic, row_bar_with_label, column_labels_list)


# In[21]:


c_l = [str(i) for i in range(1,10)]
r_l = [chr(ord('A')+i) for i in range(7)]


# In[22]:


def littlepic_processing(image):
    reverse_image = np.where(image > 200, 0, 255)
    return reverse_image

    #choose one

    clean_left_right_bounding(image, 100)
    reverse_image = np.where(image > 200, 0, 255)
    return reverse_image
        


# In[23]:


def adjust_range(spot):
    ((top, bottom), (left, right)) = spot
    top = top - 2
    bottom = bottom + 2
    left = left - 2
    right = right + 2
    return ((top, bottom), (left, right))


# In[27]:


def add_to_square(img,newsize):
    im_h ,im_w,c = img.shape
    #將圖片擴充成正方形，方便用圓周去點
    u = int((newsize-im_h)/2)
    v = int((newsize-im_w)/2)
    try:
        img = cv2.copyMakeBorder(img,u,newsize-im_h-u,v+1,newsize-im_w-v-1, cv2.BORDER_CONSTANT,value=[255,255,255])
    except:
        pass
    return img

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
        
        if d > radius-21:
            p1 = a
            p2 = b
            img[a,b] = 255
            #print(p1,p2)
        ret,thresh1 = cv2.threshold(img,175,255,cv2.THRESH_BINARY)
    return thresh1    


# In[28]:


def save_pictures_with_label(image_color ,all_contents, r_l, c_l):
    path  = './cut06/'
    if not os.path.isdir(path):
        os.mkdir(path)
    for row in all_contents:
        for column in all_contents[row]:
            spot = all_contents[row][column]
            spot = adjust_range(spot)
            image = spot_image(image_color, spot)
            image = littlepic_processing(image)
            cv2.imwrite(path+r_l[row]+c_l[column]+'.jpg', image)

#image_color = cv2.imread(file_address)
save_pictures_with_label(cutable_content_pic ,all_contents, r_l, c_l)



path  = './cut06/'
for row in all_contents:
    for column in all_contents[row]:
        image = cv2.imread(path+r_l[row]+c_l[column]+'.jpg')
        img = add_to_square(image,155)
            #for k,p in enumerate(circle_18):
            #    p1 = img[0]
            #    p2 = img[1]
            #    img[p1,p2]=255
        img = clear_circle(img)
        cv2.imwrite(path+r_l[row]+c_l[column]+'.jpg', img)

################################################################


pytesseract.pytesseract.tesseract_cmd = '.\Tesseract-OCR\\tesseract.exe'
# def revise2(code):
#     st,end =0,0
#     for i in range(len(code)-1):
#         if code[i] =='[':
#             st=i
#         elif code[i] ==']':
#             end = i            
#             break
#     str_list=[]
#     if code[end+1] is not '/':
#         str_list= list (code)
#         # nPos=str_list.index( '/' ) ##可略
#         str_list.insert(end+1,'/')
#         code= "" .join(str_list)
#     code=code[:st+1]+re.sub("\D","",code[st:end+1])+code[end:] ##re.sub->刪除非字母
#     return code

def revise(code):
    code=code.replace(" ", "",2)
    code=code.replace("\n", "",2)
    code=code.replace("RIN", "RN")
    code=code.replace("EF", "R")
    code=code.replace("WV", "W")
    code=code.replace("PIN", "PN")
    code=code.replace("AQ", "HQ")
    code=code.replace("c", "E")
    code=code.replace("MLLQXC", "EMLRQXQ")
    code=code.replace("OAS.", "Q.G.")
    code=code.replace("Z2", "2")
    code=code.replace("Z", "2")
    code=code.replace("2Z", "2")

    if len(code)>0:
        if code[-1]=="I":
            code=code[:len(code)-1]+"1"


    # code=code.replace("S", "5",1)
    if len(code)>1 :
        if code[0]=='0':
            code='O'+code[1:]
        if code[0]=='1':
            code='I'+code[1:]
    # if i<8 and len(code)==1 :
    #     code=code.replace("]", "7",1)
    #     code=code.replace("A", "4",1)

    # if '[' in code:
    #     code=code.replace("O", "0",4)
    #     if code.count('[')==2:
    #         code=revise2(code)
    #     elif code.count('[')==1 and code.count(']')==0 :
    #         code=code+"]"
    return code

def idtfy(ws,wb):
    path = "./cut06/"###改
    dirs = os.listdir( path )
    for filename in dirs:
        name=filename.split('.', 1 )             
        img = cv2.imread(path + filename)
        img = cv2.copyMakeBorder(img,10,10,10,10, cv2.BORDER_CONSTANT,value=[255,255,255])#### 加白框
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
    wb.save('Team_025_06.xlsx')###改




write_xls()

