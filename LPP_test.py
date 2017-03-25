#import matplotlib.pyplot as plt
from moviepy.video.io.VideoFileClip import VideoFileClip
#from lpproj import LocalityPreservingProjection 
from sklearn import manifold
from sklearn.decomposition import PCA
from PIL import Image
from vgg import vgg16
import tensorflow as tf
import numpy as np
import math
import time
import cv2


def test_vgg(temp_img):
    with tf.device('/cpu:0'):
        vgg = vgg16.Vgg16()
        with tf.Session() as sess:
            images = tf.placeholder("float", [1, 224, 224, 3])
            with tf.name_scope("content_vgg"):
                vgg.build(images)
    temp_img= temp_img.resize((224, 224))
    norm = np.divide(np.array(temp_img), 255)
    norm = np.reshape(norm, [1,224, 224, 3])        
    feed_dict = {images: norm}
    return sess.run(vgg.fc6, feed_dict=feed_dict)



count=0
max_frames=110
clip = VideoFileClip("./video_test2.mp4")
width=1280
hight=720
temp_hight=250
temp_width=130
width_len=math.floor(temp_width/3.0)
hight_len=math.floor(temp_hight/3.0)
width_step=math.floor(width/width_len)
hight_step=math.floor(width/hight_len)
bg=Image.open("./black.jpg")
#lpp = LocalityPreservingProjection(n_components=2)

for frame in clip.iter_frames():
    count+=1
    if count==max_frames:
        break
    start=time.time()
    number=0
    blurred = cv2.GaussianBlur(frame, (3, 3), 0)
    img=Image.fromarray(blurred)
    result=[]
    images=[]
    bg_copy=bg.copy()
    for i in range(width_step):
        for j in range(hight_step):
            x1=i*width_len
            y1=j*hight_len
            x2=x1+temp_width
            y2=y1+temp_hight
            if x1>width:
                x1=width
            if y1>hight:
                y1=hight
            if x2>width:
                x2=width
            if y2>hight:
                y2=hight
            temp_img=img.crop((x1,y1,x2,y2))
            images.append(temp_img.resize((30,30)))
            temp_img= temp_img.resize((224, 224))
            hls = cv2.cvtColor(np.array(temp_img), cv2.COLOR_RGB2HLS)
            result.append(list(np.reshape(hls[:,:,0:2], [224*224*2])))
            number+=1
    print(np.array(result).shape)
    #print("Computing PCA")
    #pca=PCA()
    #result=pca.fit(np.array(result))
    print("Computing LLE embedding")
    X_2D, err = manifold.locally_linear_embedding(np.array(result), n_neighbors=12,
                                             n_components=2)
    print("Done. Reconstruction error: %g" % err)
    #X_2D = lpp.fit_transform(np.array(result))
    end=time.time()
    print("time:",end-start)
    
    for i in range(len(images)):
        x1=math.floor((X_2D[i][0]+1)*width/2.0)
        y1=math.floor((X_2D[i][1]+1)*hight/2.0)
        print(X_2D[i][0],X_2D[i][1],x1,y1)
        bg_copy.paste(images[i],(x1,y1))
    bg_copy.save("./test_out/catch_distributed/%06d.jpg"%count,bg.format)
    a=input("test")
    if (count%10)==0:
        print(count)
    
    
    
#img=Image.fromarray(frame)
#img.save(name,img.format)
    


    

    
