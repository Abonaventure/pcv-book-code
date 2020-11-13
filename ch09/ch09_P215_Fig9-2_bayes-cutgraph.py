# from scipy.misc import imresize  ##已被弃用，用from PIL import Image替代
import graphcut
from PIL import Image
from pylab import *

# 添加中文字体支持
from matplotlib.font_manager import FontProperties
font = FontProperties(fname=r"/System/Library/Fonts/PingFang.ttc", size=14)

# 读入图像
im = array(Image.open("../data/empire.jpg"))
h,w = im.shape[:2]
print(h,w)
scale = 0.05  #  scale = 0.265 ，scale*scale ~= 0.07跑起来非常慢,scale=0.05代码跑通比较快
num_px = int(w * scale)
num_py = int(h * scale) 
# imresize(im, 0.07,interp='bilinear')  ##imresize被scipy.misc弃用，用PIL库中的resize替代
im = array(Image.fromarray(im).resize((num_px,num_py),Image.BILINEAR))
size = im.shape[:2]
print(size)
rm = im

# 添加两个矩形训练区 
labels = zeros(size)
labels[3:18, 3:18] = -1
labels[-18:-3, -18:-3] = 1
# print(labels.size)
print("labels finish")

# 创建训练图
g = graphcut.build_bayes_graph(im, labels, kappa=1)
print("build_bayes_graph finish")

# 得到分割图
res = graphcut.cut_graph(g, size)
print("cut_graph finish")

#显示标记图
fig = figure()
subplot(131)
graphcut.show_labeling(im, labels)
gray()
title(u'标记图', fontproperties=font)
axis('off')

# #显示训练图
subplot(132)
imshow(rm) 
contour(labels,[-0.5,0.5],colors='blue') 
contour(labels,[0.5,1],colors='yellow') 
# gray()
title(u'训练图', fontproperties=font)
axis('off')

#显示分割图
subplot(133)
imshow(res)
gray()
title(u'分割图', fontproperties=font)
axis('off')
show()

# 保存figure中的灰度图像和积分图像
fig.savefig("../images/ch09/ch10_P215_Fig-2_bayes-cutgraph.jpg")

print("finish")