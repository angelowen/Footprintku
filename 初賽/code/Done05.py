#!/usr/bin/env python
# coding: utf-8
import numpy as np
import cv2,csv
import pytesseract,os
import time
from PIL import Image,ImageDraw
import matplotlib.pyplot as plt
from collections import defaultdict
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# all function
def range_of_entire_image(image):
    vertical, horizontial = image.shape
    top = 0
    bottom = vertical-1
    left = 0
    right = horizontial -1
    return ((top, bottom), (left, right))

def extract_peek_ranges_from_array(array_vals, minimun_val=10, minimun_range=2):
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

# 可以輸出指定範圍的圖片
def crop_range_of_image(image, range_tuple):
    ((top, bottom), (left, right)) = range_tuple
    return image[top : bottom, left : right]

def ranged_horizontal(image, range_tuple = None):
    if range_tuple == None:
        range_tuple = range_of_entire_image(image)
    horizontal_sum = np.sum(crop_range_of_image(image, range_tuple), axis=1)
    ranges = extract_peek_ranges_from_array(horizontal_sum)
    start_point = range_tuple[0][0]; 
    ranges2 = [(start_point + start, start_point + end) for start, end in ranges]
    return [(h_ranges, range_tuple[1]) for h_ranges in ranges2]

def ranged_vertical(image, range_tuple = None):
    if range_tuple == None:
        range_tuple = range_of_entire_image(image)
    vertical_sum = np.sum(crop_range_of_image(image, range_tuple), axis=0)
    ranges = extract_peek_ranges_from_array(vertical_sum)
    start_point = range_tuple[1][0]; 
    ranges2 = [(start_point + start, start_point + end) for start, end in ranges]
    return [(range_tuple[0], v_ranges) for v_ranges in ranges2]

def show_ranges_on_image(image, ranges_list):
    vision = np.copy(image)
    for i, peek_range in enumerate(ranges_list):
        x = 0
        y = peek_range[0]
        w = vision.shape[1]
        h = peek_range[1] - y
        pt1 = (x, y)
        pt2 = (x + w, y + h)
        #cv2.rectangle(影像, 頂點座標, 對向頂點座標, 顏色, 線條寬度)
        cv2.rectangle(vision, pt1, pt2, 255)

    cv2.imshow('ranges image', line_seg_adaptive_threshold)
    cv2.waitKey(0)
    return
   
# pop出最窄的範圍 可以用來找出 label 部份的圖像
# axis 輸入要比較的維度
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
            
def column_content_bonder(image, content_range, column_label):
    percent = 0.02
    ((_, _), (label_left, label_right)) = column_label
    mid = round((label_left + label_right) / 2)
    lsls = np.sum(crop_range_of_image(image, content_range), axis=0)
    left = right = (mid - content_range[1][0])
    max_sum = content_range[0][1] - content_range[0][0]
    max_sum = max_sum * 255
    lower = max_sum * percent * 0.6
    upper = max_sum * (1 - percent)
    # print("range:", lower, "to", upper)
    while lower < lsls[left] < upper and left > 0 :
        left = left - 1
        
    while lower < lsls[right] < upper and right < lsls.size:
        right = right + 1
    # print(lsls[left : right])    
    return (left + content_range[1][0], right + content_range[1][0])

def column_content_ranges(image, peeks, column_labels):
    column_contents = {}
    for content_range in peeks:
        for i, label in enumerate(column_labels) :
            if  content_range[1][0] < (label[1][0] + label[1][1]) / 2 < content_range[1][1] :
                LR = column_content_bonder(image, content_range, label)
                column_contents[i] = (content_range[0],LR)
    
    return column_contents
        
def row_content_bonder(image, content_range, row_label):
    percent = 0.02
    mid = round((row_label[0][0] + row_label[0][1] ) / 2)
    lsls = np.sum(crop_range_of_image(image, content_range), axis=1)
    left = mid - content_range[0][0]
    right = mid - content_range[0][0]
    max_sum = content_range[1][1] - content_range[1][0]
    max_sum = max_sum * 255
    lower = max_sum * percent * 0.6
    upper = max_sum * (1 - percent)
    while lower < lsls[left] < upper and left > 0 :
        left = left - 1
        
    while lower < lsls[right] < upper and right < lsls.size:
        right = right + 1
        
    return (left + content_range[0][0], right + content_range[0][0])




def littlepic2chars(polar_image):
    range_tuple = range_of_entire_image(polar_image)
    
    horizontal_sum = np.sum(crop_range_of_image(polar_image, range_tuple), axis=1)
    ranges = saperate_by_space(horizontal_sum,1,10)
    start_point = 0
    ranges2 = [(start_point + start, start_point + end) for start, end in ranges]
    horizontal_bars =  [(h_ranges, range_tuple[1]) for h_ranges in ranges2]
    
    chars = []
    for bar in horizontal_bars:
        vertical_sum = np.sum(crop_range_of_image(polar_image, bar), axis=0)
        ranges = saperate_by_space(vertical_sum,1,10)
        start_point = 0
        ranges2 = [(start_point + start, start_point + end) for start, end in ranges]
        for v_ranges in ranges2:
            chars.append((bar[0], v_ranges))
            
    return chars

# clean the circle
def clean_left_right_bounding(polar_image):
    for row in polar_image:
        cleaner = False;
        for i,pixel in enumerate(row):
            if cleaner == True and pixel < 200 :
                cleaner = False;
                break;
            if pixel > 200 :
                cleaner = True;
            if cleaner == True :
                row[i] = 0;

        for i,pixel in reversed(list(enumerate(row))):
            if cleaner == True and pixel < 200 :
                cleaner = False;
                break;
            if pixel > 200 :
                cleaner = True;
            if cleaner == True :
                row[i] = 0;
    return

# one of this can clean the box

def clean_top_bottom_bounding(polar_image):
    bottom, right = polar_image.shape
    for column in range(right-1):
        cleaner = False;
        for row in range(bottom-1):
            pixel = polar_image[row][column]
            if cleaner == True and pixel < 200 :
                cleaner = False;
                break;
            if pixel > 200 :
                cleaner = True;
            if cleaner == True :
                polar_image[row][column] = 0;

        for row in reversed(range(bottom-1)):
            pixel = polar_image[row][column]
            if cleaner == True and pixel < 200 :
                cleaner = False;
                break;
            if pixel > 200 :
                cleaner = True;
            if cleaner == True :
                polar_image[row][column] = 0;
    return


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





def ranged_horizontal_by_space(image, range_tuple, space_len):
    if range_tuple == None:
        range_tuple = range_of_entire_image(image)
    horizontal_sum = np.sum(crop_range_of_image(image, range_tuple), axis=1)
    #print(horizontal_sum)
    #show(crop_range_of_image(image, range_tuple)) # for test
    ranges = saperate_by_space(horizontal_sum,space_len,300)
    start_point = range_tuple[0][0]; 
    ranges2 = [(start_point + start, start_point + end+1) for start, end in ranges]
    return [(h_ranges, range_tuple[1]) for h_ranges in ranges2]




def ranged_vertical_by_space( image, range_tuple , space_len):
    if range_tuple == None:
        range_tuple = range_of_entire_image(image)
    horizontal_sum = np.sum(crop_range_of_image(image, range_tuple), axis=0)
    ranges = saperate_by_space(horizontal_sum,space_len,10)
    start_point = range_tuple[1][0]; 
    ranges2 = [(start_point + start, start_point + end+1) for start, end in ranges]
    return [(range_tuple[0], v_ranges) for v_ranges in ranges2]









# image_prepare
image_color = cv2.imread("./FPK_05.jpg")
image = cv2.cvtColor(image_color, cv2.COLOR_BGR2GRAY)
polar_image = cv2.adaptiveThreshold(
    image,
    255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY_INV, 11, 2)
# make image input success
polar_image.shape



horizontal_cut = ranged_horizontal(polar_image)
column_label = horizontal_cut.pop(index_of_min_range(horizontal_cut, 0))
clear_muti_lable(horizontal_cut, column_label, 0)

vertical_cut = ranged_vertical(polar_image, horizontal_cut[0])
row_label = vertical_cut.pop(index_of_min_range(vertical_cut, 1))
clear_muti_lable(vertical_cut, row_label, 1)

content_list = vertical_cut




list_row_label = ranged_horizontal(polar_image, row_label)
list_column_label = ranged_vertical_by_space(polar_image, column_label,20)




clean_left_right_bounding(crop_range_of_image(polar_image, content_list[0]))




horizontal_bars = ranged_horizontal_by_space(polar_image, content_list[0], 2)




#match bar and row label
di = {}
for i, label in enumerate(list_row_label):
    mid = (label[0][0]+label[0][1])/2 
    for bar in horizontal_bars:
        (top, bottom),(_,_) = bar
        if(top < mid < bottom):
            di[i] = bar
row_bar_with_label = di



try:
    os.mkdir(str('./column_labels/'))
    for column, chars in enumerate(list_column_label):
        cv2.imwrite('./column_labels/column'+str(column)+'.jpg', crop_range_of_image(polar_image, chars))    

    os.mkdir(str('./row_labels/'))
    for row, chars in enumerate(list_row_label):
        cv2.imwrite('./row_labels/column'+str(row)+'.jpg', crop_range_of_image(polar_image, chars))  
except:
    pass




all_contents = {}
for row in row_bar_with_label:
    littlepics = ranged_vertical_by_space(polar_image, row_bar_with_label[row], 3)
    di = {}
    for strpic in littlepics:
        left = strpic[1][0]
        right = strpic[1][1]
        for i_column, column_label in enumerate(list_column_label) :
            label_mid = (column_label[1][0] + column_label[1][1])/2
            if  left < label_mid < right :
                di[i_column] = strpic
                
    all_contents[row] = di
            




c_l = [str(x+1) for x in range(9)]

r_l = [chr(ord('A')+x) for x in range(12)]

r_l.pop(8)



img = image_color
path  = './5string/'
if not os.path.isdir(path):
    os.mkdir(path)
    
for row in all_contents:
    for column in all_contents[row]:
        spot = all_contents[row][column]
        part = ranged_horizontal_by_space(polar_image, spot, 1)
        O = crop_range_of_image(polar_image, part[0])
        headline = O[-2:]
        leng = headline.sum()
        if leng > 3000 :
            key = '1'
        else:
            key = '0'
            
        #print(key," ",row," ",column)
        #cv2.imwrite(path+key+r_l[row]+c_l[column]+'.jpg', crop_range_of_image(polar_image, part[1]))
        
        try:
            cv2.imwrite(path+str(key)+r_l[row]+c_l[column]+'.jpg', crop_range_of_image(img, part[1]))
            #cv2.imshow(path+key+r_l[row]+c_l[column]+'.jpg', crop_range_of_image(polar_image, part[1]))
            #cv2.waitKey(0)
            #cv2.destroyAllWindows()
        except:
            pass


################################
path = "./5string/"
dirs = os.listdir( path )
with open('FPK_05.csv','w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["NUMBER","CONTENT"])
    for filename in dirs:
        flag=0
        img = cv2.imread(path+filename)
        a = cv2.copyMakeBorder(img,4,4,4,4, cv2.BORDER_CONSTANT,value=[255,255,255])
        im_h,im_w,tmp=a.shape
        # print(a.shape)
        a=cv2.resize(a,(im_w*10,im_h*10))
        # cv2.imshow("img",a)
        # cv2.waitKey(0)
        code=pytesseract.image_to_string(a,lang='eng',config='-psm 4 ')#,lang='eng',config='-psm 7 digits'
        if len(code)>0 and code[0]=='0':
            code='O'+code[1:len(code)]
        if len(code)>2 and (code[len(code)-1]=='O' or code[len(code)-1]=='o') :
            code=code[0:len(code)-1]+'0'
        for i in range(0,len(code)):
            if code[i]=='v':
                code=code[0:i]+'Y'+code[i+1:len(code)]
        for i in range(0,len(code)):
            if str.isdigit(code[i])==False and str.isalpha(code[i])==False:
                code1=code[0:i]
                code2=code[i+1:]
                code1+=code2
                flag=1    
        name = [  str(x) for x in filename.split( '.' )]
        if(name[0][0]=='1'):
            l=len(code)
            k=0
            while(k<l):
                code=code[0:k+1]+'\\'+code[k+1:]
                l=len(code)
                k+=2
        if flag:
            print(name[0][1:],code1)
            writer.writerow([name[0][1:],code1])
        else:
            print(name[0][1:],code)
            writer.writerow([name[0][1:],code])
            
            




