# from scipy.misc import imresize
# from PCV.tools import graphcut
import graphcut
from PIL import Image
from pylab import *
# import numpy as np
# # import pylab as pl
# import matplotlib.pyplot as plt


# 添加中文字体支持
from matplotlib.font_manager import FontProperties
font = FontProperties(fname=r"/System/Library/Fonts/PingFang.ttc", size=14)

def create_msr_labels(m, lasso=False):
    """ Create label matrix for training from
    user annotations. """
    # rim = im.reshape((-1,2))
    # m = m.convert("L")
    size = m.shape[:2]
    # m = Image.fromarray(m.astype('uint8')).convert("L")
    # size = m.shape[:2]
    labels = zeros(size)
    # background
    labels[m == 0] = -1
    labels[m == 64] = -1
    # foreground
    if lasso:
        labels[m == 255] = 1
    else:
        labels[m == 128] = 1
    return labels

# load image and annotation map
im = array(Image.open('../data/book_perspective.JPG'))
m = array(Image.open('../data/book_perspective.bmp').convert('L'))

# resize
# scale = 0.32
scale = 0.05 #  scale = 0.32 ，scale*scale ~= 0.1跑起来非常慢,scale=0.05代码跑通比较快
# im = imresize(im, scale, interp='bilinear')
# m = imresize(m, scale, interp='nearest')
h1,w1 = im.shape[:2]
h2,w2 = m.shape[:2]
print(h1,w1)
print(h2,w2)
# num_px = int(h * np.sqrt(0.07))
# num_py = int(w * np.sqrt(0.07))
px1 = int(w1 * scale)
py1 = int(h1 * scale) 
px2 = int(w2 * scale)
py2 = int(h2 * scale)
# imresize(im, 0.07,interp='bilinear')  ##imresize被scipy.misc弃用，用PIL库中的resize替代
im = array(Image.fromarray(im).resize((px1,py1),Image.BILINEAR))
m = array(Image.fromarray(m).resize((px2,py2),Image.NEAREST))
oim = im
print(im.shape[:2])
print(m.shape[:2])
# create training labels
labels = create_msr_labels(m, False)
print('labels finish')
# build graph using annotations
g = graphcut.build_bayes_graph(im, labels, kappa=2)
print('build_bayes_graph finish')
# cut graph
res = graphcut.cut_graph(g, im.shape[:2])
print('cut_graph finish')
# remove parts in background
res[m == 0] = 1
res[m == 64] = 1
# labels[m == 0] = 1
# labels[m == 64] = 1

# plot original image 
fig = figure()
subplot(121)
imshow(im)
gray()
title(u'原始图', fontproperties=font)
axis('off')

#plot the result
subplot(122)
imshow(res)
gray()
xticks([])
yticks([])
title(u'分割图', fontproperties=font)
axis('off')

show()
fig.savefig('../images/ch09/labelplot.pdf')

print('finish')

