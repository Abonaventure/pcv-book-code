import pixellib
from pixellib.tune_bg import alter_bg
import cv2


change_bg = alter_bg()
change_bg.load_pascalvoc_model("deeplabv3_xception_tf_dim_ordering_tf_kernels.h5")

capture = cv2.VideoCapture(0)
while True:
    ret, frame = capture.read()
    output = change_bg.change_frame_img(frame,b_image_path = "./Images/flowers.jpg")  ###将视频背景换成flowers图片
    # output = change_bg.color_frame(frame, colors = (255, 255, 255)) ###将视频背景换成(255, 255, 255)彩色图片
    # output = change_bg.gray_frame(frame)  ###将视频背景换成灰色图片
    # output = change_bg.blur_frame(frame, extreme = True)###将视频背景换成模糊背景
    cv2.imshow("frame", output)
    if  cv2.waitKey(25) & 0xff == ord('q'):
        break
 
