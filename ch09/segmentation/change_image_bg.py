import pixellib
from pixellib.tune_bg import alter_bg
import cv2


change_bg = alter_bg()
change_bg.load_pascalvoc_model("deeplabv3_xception_tf_dim_ordering_tf_kernels.h5")
# change_bg.change_bg_img(f_image_path = "sample.jpg",b_image_path = "background.jpg", output_image_name="new_img.jpg")
output = change_bg.change_bg_img(f_image_path = "./Images/p1.jpg",b_image_path = "./Images/flowers.jpg", output_image_name="flowers_bg.jpg")
cv2.imwrite("img.jpg", output)

change_bg = alter_bg()
change_bg.load_pascalvoc_model("deeplabv3_xception_tf_dim_ordering_tf_kernels.h5")
# change_bg.color_bg("sample.jpg", colors = (0,0,255), output_image_name="colored_bg.jpg")
output = change_bg.color_bg("./Images/p1.jpg", colors = (0,0,255), output_image_name="colored_bg.jpg")
cv2.imwrite("img.jpg", output)

change_bg = alter_bg()
change_bg.load_pascalvoc_model("deeplabv3_xception_tf_dim_ordering_tf_kernels.h5")
# change_bg.gray_bg("sample.jpg",output_image_name="gray_img.jpg")
output = change_bg.gray_bg("./Images/p1.jpg",output_image_name="gray_bg.jpg")
cv2.imwrite("img.jpg", output)

hange_bg = alter_bg()
change_bg.load_pascalvoc_model("deeplabv3_xception_tf_dim_ordering_tf_kernels.h5")
# change_bg.blur_bg("sample2.jpg", low = True, output_image_name="blur_img.jpg")
output = change_bg.blur_bg("./Images/p1.jpg", low = True, output_image_name="blur_bg.jpg")
cv2.imwrite("img.jpg", output)


from pylab import *
from PIL import Image
# 添加中文字体支持
from matplotlib.font_manager import FontProperties
font = FontProperties(fname=r"/System/Library/Fonts/PingFang.ttc", size=14)

figure()
subplot(231)
imshow(array(Image.open("./Images/p1.jpg")))
title(u'original p1', fontproperties=font)
axis("off")

subplot(232)
imshow(array(Image.open("./Images/flowers.jpg")))
title(u'original flowers', fontproperties=font)
axis("off")

subplot(233)
imshow(array(Image.open("flowers_bg.jpg")))
title(u'flowers_bg', fontproperties=font)
axis("off")

subplot(234)
imshow(array(Image.open("colored_bg.jpg")))
title(u'colored_bg', fontproperties=font)
axis("off")

subplot(235)
imshow(array(Image.open("gray_bg.jpg")))
title(u'gray_bg', fontproperties=font)
axis("off")

subplot(236)
imshow(array(Image.open("blur_bg.jpg")))
title(u'blur_bg', fontproperties=font)
axis("off")

show()

