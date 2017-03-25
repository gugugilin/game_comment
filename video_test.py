from moviepy.video.io.VideoFileClip import VideoFileClip
import catch_players as catch
from PIL import Image
import Poslect as pl
import cv2
import time
import numpy as np


p1=catch.catch_players(thresdhold=0.15,show_console=False)
poslect=pl.Poslect()
count=0
max_frames=110
clip = VideoFileClip("./video_test2.mp4")
target=[]
input_data=[]
observe_len=3
batch_size=5
mask=np.array([[1/1280.0,0,0,0],[0,1/720.0,0,0],[0,0,1/1280.0,0],[0,0,0,1/720.0]])
start=time.time()
for frame in clip.iter_frames():
    if count==max_frames:
        break
    name="./test_out/origin/%06d.jpg"%count
    img=Image.fromarray(frame)
    #img.save(name,img.format)
    result,_,_,_,_=p1.test(img=frame,save_draw=False)
    result=np.array(result)
    print(result.shape)
    target.append(list(np.reshape(np.dot(result,mask),[40])))
    blurred = cv2.GaussianBlur(frame, (5, 5), 0)
    gray = cv2.cvtColor(blurred, cv2.COLOR_RGB2GRAY)
    img_gray=Image.fromarray(gray)
    input_data.append(list(np.reshape(np.array(img_gray.resize((264,264),Image.BILINEAR)),[264*264])))
    count+=1
    if count>=batch_size:
        start1=time.time()
        print("traing:",count," (start)")
        print(np.array(input_data).shape)
        poslect.fit(input_data,target,observe_len,50)
        end1=time.time()
        print("traing:",count," (end) time:",end1-start1)
        del input_data[0]
        del target[0]
end=time.time()
print("total traing end",end-start)
"""
name="./test_out/filter_result/%06d.jpg"%count
    name1="./test_out/mask1/%06d.jpg"%count
    name2="./test_out/mask2/%06d.jpg"%count
    name3="./test_out/detector/%06d.jpg"%count
    img.save(name,img.format)
    img1.save(name1,img1.format)
    img2.save(name2,img2.format)
    img3.save(name3,img3.format)
"""

    

    