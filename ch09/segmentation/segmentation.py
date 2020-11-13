##step1.导入pixellib模块
import pixellib
from pixellib.semantic import semantic_segmentation
from pixellib.instance import instance_segmentation
##step2.创建用于执行语义分割的类实例
segment_image = semantic_segmentation()
##step3.调用load_pascalvoc_model()函数加载在Pascal voc上训练的Xception模型
segment_image.load_pascalvoc_model("deeplabv3_xception_tf_dim_ordering_tf_kernels.h5")
##step4.调用segmentAsPascalvoc()函数对图像进行分割
##segment_image.segmentAsPascalvoc("path_to_image", output_image_name = "path_to_output_image")
segment_image.segmentAsPascalvoc("./Images/sample1.jpg", output_image_name = "image_new1.jpg", overlay = True)


segment_image = instance_segmentation()

segment_image.load_model("mask_rcnn_coco.h5")

##segment_image.segmentImage("path_to_image", output_image_name = "output_image_path")

segment_image.segmentImage("./Images/sample2.jpg", output_image_name = "image_new2.jpg", show_bboxes = True)

from pylab import *
from PIL import Image
# 添加中文字体支持
from matplotlib.font_manager import FontProperties
font = FontProperties(fname=r"/System/Library/Fonts/PingFang.ttc", size=14)

figure()
subplot(221)
imshow(array(Image.open("./Images/sample1.jpg")))
title(u'原图1', fontproperties=font)
axis("off")

subplot(222)
imshow(array(Image.open("image_new1.jpg")))
title(u'原图1语义分割图', fontproperties=font)
axis("off")

subplot(223)
imshow(array(Image.open("./Images/sample2.jpg")))
title(u'原图2', fontproperties=font)
axis("off")

subplot(224)
imshow(array(Image.open("image_new2.jpg")))
title(u'原图2实例分割图', fontproperties=font)
axis("off")

show()