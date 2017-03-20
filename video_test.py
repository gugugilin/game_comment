from moviepy.video.io.VideoFileClip import VideoFileClip
import catch_players as catch
from PIL import Image


p1=catch.catch_players(thresdhold=0.08,show_console=False)
result=[]
count=10
max_frames=110+count
clip = VideoFileClip("./video_test2.mp4")
for frame in clip.iter_frames():
    if (count%10)==0:
        print(count)
    if count==max_frames:
        break
    name="./test_out/origin/%06d.jpg"%count
    img=Image.fromarray(frame)
    img.save(name,img.format)
    _,img,img1,img2,img3=p1.test(img=frame)
    name="./test_out/filter_result/%06d.jpg"%count
    name1="./test_out/mask1/%06d.jpg"%count
    name2="./test_out/mask2/%06d.jpg"%count
    name3="./test_out/detector/%06d.jpg"%count
    img.save(name,img.format)
    img1.save(name1,img1.format)
    img2.save(name2,img2.format)
    img3.save(name3,img3.format)
    count+=1
    


    

    