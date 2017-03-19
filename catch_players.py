from PIL import Image
from net.build import TFNet
import numpy as np
import math
import cv2
import json
import time


def catch_hsv(img,hlsdown,hlsup,width,hight,threadhold_area,save_name,color,limt_mask,bg):
    img_copy=img.copy()
    black_image=bg.copy()
    width=math.floor(width/2)
    hight=math.floor(hight/2)
    #CV_BGR2HLS
    #blurred = cv2.GaussianBlur(np.array(img_copy), (3, 3), 0)
    hls = cv2.cvtColor(np.array(img_copy), cv2.COLOR_RGB2HLS)
    mask = cv2.inRange(hls, hlsdown, hlsup)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=4)
    
    # find contours in the mask and initialize the current(x, y) center 
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[-2]
    count=0
    baise=40
    datas=[]
    if len(cnts) > 0:
        for cnt in cnts:
            x,y,w,h = cv2.boundingRect(cnt)
            if x+width<=1280 and y+hight+baise<=720:                
                if w*h>=threadhold_area and limt_mask[y+hight+baise][x]==1 and limt_mask[y+hight+baise][x+width]==1:
                    datas.append([x-width,y-hight+baise,x+width,y+hight+baise,x+baise,y+baise])
                    black_image.paste(img.crop((x-width,y-hight+baise,x+width,y+hight+baise)),(x-width,y-hight+baise))
                    count+=1
    else:
        print("not target obj")
    print(count)
    black_image.save(save_name,img.format)
    return np.array(black_image),datas

def match_person(datas,mask,filters,r):
    if datas is not None:
        max_pro=[10000000 for i in range(len(filters))]
        record_point=[[0,0,0,0] for i in range(len(filters))]
        for data in datas:
            bottomright=[data['bottomright']['x'],data['bottomright']['y']]
            if mask[bottomright[1]][bottomright[0]]==1 :
                topleft=[data['topleft']['x'],data['topleft']['y']]
                max_area=0
                position=-1
                temp_point=[0,0,0,0]
                i=0
                for filter_temp in filters:
                    width=((bottomright[0]-topleft[0])+(filter_temp[2]-filter_temp[0]))-(max(filter_temp[2],bottomright[0])-min(filter_temp[0],topleft[0]))
                    hight=((bottomright[1]-topleft[1])+(filter_temp[3]-filter_temp[1]))-(max(filter_temp[3],bottomright[1])-min(filter_temp[1],topleft[1]))
                    if width>0 and hight>0:
                        if (width*hight)>=r*((filter_temp[3]-filter_temp[1])*(filter_temp[2]-filter_temp[0])):
                            if max_area <width*hight:
                                max_area=width*hight
                                position=i
                                x=filter_temp[4]
                                y=filter_temp[5]
                                p1=[max(filter_temp[0],topleft[0]),max(filter_temp[1],topleft[1]),min(filter_temp[2],bottomright[0]),min(filter_temp[3],bottomright[1])]
                                temp_point=[p1[0]+math.floor((-p1[0]-p1[2]+2*x)/4),p1[1]+math.floor((-p1[1]-p1[3]+2*y)/4),p1[2]+math.floor((-p1[0]-p1[2]+2*x)/4),p1[3]+math.floor((-p1[1]-p1[3]+2*y)/4)]
                    i+=1
                if position>=0 and max_pro[position] > max_area:
                    #print(position,max_area,temp_point)
                    record_point[position]=temp_point
                    max_pro[position]=max_area
    return record_point
def draw_person(datas,color,img,save_name):
    for data in datas:
        bottomright=(data['bottomright']['x'],data['bottomright']['y'])
        topleft=(data['topleft']['x'],data['topleft']['y'])
        img=cv2.rectangle(np.array(img),topleft,bottomright,color,2) 
        img=Image.fromarray(img)
    img.save(save_name,img.format)
    return img
def draw_box(datas,color,img,save_name):
    if datas is not None:
        for data in datas:
            topleft=(data[0],data[1])
            bottomright=(data[2],data[3])
            img=cv2.rectangle(np.array(img),topleft,bottomright,color,2) 
            img=Image.fromarray(img)
        img.save(save_name,img.format)
    return img


def catch_basketball_court(img,hlsdown,hlsup,threadhold):
    img_copy=img.copy()
    img_copy = cv2.GaussianBlur(np.array(img_copy), (3, 3), 0)
    hls = cv2.cvtColor(img_copy, cv2.COLOR_RGB2HLS)
    G = cv2.inRange(hls, hlsdown, hlsup)
    mask = cv2.inRange(hls, hlsdown, hlsup)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=4)
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[-2]
    
    if len(cnts) > 0:
        left_point=10000000
        right_point=0
        top_point=10000000
        down_point=0
        for cnt in cnts:
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            width=abs(box[2][0]-box[0][0])
            hight=abs(box[2][1]-box[0][1])
            if width*hight>=threadhold:
                if left_point>box[0][0]:
                    left_point=box[0][0]
                if  right_point<box[2][0]:
                    right_point=box[2][0]
                if  top_point>box[2][1]:
                    top_point=box[2][1]
                if  down_point<box[0][1]:
                    down_point=box[0][1]
        data=np.array([[[left_point,down_point],[left_point,top_point],[right_point,top_point],[right_point,down_point]]])
        mask=[  [0 for i in range(1280)]   for j in range(720)   ]
        for i in range(top_point,down_point):
            for j in range(left_point,right_point):
                mask[i][j]=1
    return mask


def test(image_path):
    options = {"model": "cfg/yolo-voc.cfg", "load": "bin/yolo-voc.weights", "threshold": 0.2}

    tfnet = TFNet(options)
    img=Image.open(image_path)

    bg=Image.open("./black.jpg")
    print("start test")

    start=time.time()
    datas=tfnet.return_predict(np.array(img))
    #draw_person(datas,(255,0,0),img,"./4.jpg")
    end=time.time()
    print("yolo time:" ,end-start)
    ans=[]
    mask=catch_basketball_court(img,(10,155,115),(20,216,153),1000)
    _,filters=catch_hsv(img,(107,71,178),(110,128,255),110,250,500,"./1.jpg",(0,255,0),mask,bg)
    ans.extend(match_person(datas,mask,filters,0.3))
    _,filters=catch_hsv(img,(125,160,0),(170,255,255),110,250,500,"./2.jpg",(0,0,255),mask,bg)
    ans.extend(match_person(datas,mask,filters,0.3))
    #img=draw_box(result1,(0,255,0),img,"./3.jpg")
    #draw_box(result2,(0,0,255),img,"./3.jpg")
    return ans

