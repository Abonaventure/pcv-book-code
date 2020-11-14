# coding=utf-8
"""
Function: figure 6.4
    Clustering of pixels based on their color value using k-means.
"""
from scipy.cluster.vq import *
# from scipy.misc import imresize
from pylab import *
from PIL import Image

# 添加中文字体支持
from matplotlib.font_manager import FontProperties
# font = FontProperties(fname=r"c:\windows\fonts\SimSun.ttc", size=14)
font = FontProperties(fname="/System/Library/Fonts/PingFang.ttc", size=14)

def pix_cluster(infile,steps):
	# steps = 100  # image is divided in steps*steps region
	# infile = '../data/empire.jpg'
	im = array(Image.open(infile))
	dx = im.shape[0] // steps
	dy = im.shape[1] // steps
	# compute color features for each region
	features = []
	for x in range(steps):
	    for y in range(steps):
	        R = mean(im[x * dx:(x + 1) * dx, y * dy:(y + 1) * dy, 0])
	        G = mean(im[x * dx:(x + 1) * dx, y * dy:(y + 1) * dy, 1])
	        B = mean(im[x * dx:(x + 1) * dx, y * dy:(y + 1) * dy, 2])
	        features.append([R, G, B])
	features = array(features, 'f')     # make into array
	# cluster
	centroids, variance = kmeans(features, 3)
	code, distance = vq(features, centroids)
	# create image with cluster labels
	codeim = code.reshape(steps, steps)
	h,w = im.shape[:2]
	# codeim = imresize(codeim, im.shape[:2], 'nearest')
	codeim = array(Image.fromarray(codeim).resize((w,h),Image.NEAREST))

	return im,codeim


infile1 = '../data/empire.jpg'
infile2 = '../data/boy_on_hill.jpg'
steps1 = 50  # image is divided in steps*steps region
steps2 = 100  # image is divided in steps*steps region
im11,codeim11 =pix_cluster(infile1,steps1)
im12,codeim12 =pix_cluster(infile1,steps2)
im21,codeim21 =pix_cluster(infile2,steps1)
im22,codeim22 =pix_cluster(infile2,steps2)


figure()

subplot(231)
title(u'原图1', fontproperties=font)
#ax1.set_title('Image')
axis('off')
imshow(im11)

subplot(232)
title(u'50×50窗格聚类', fontproperties=font)
#ax2.set_title('Image after clustering')
axis('off')
imshow(codeim11)

subplot(233)
title(u'100×100窗格聚类', fontproperties=font)
#ax2.set_title('Image after clustering')
axis('off')
imshow(codeim12)

subplot(234)
title(u'原图2', fontproperties=font)
#ax1.set_title('Image')
axis('off')
imshow(im22)

subplot(235)
title(u'50×50窗格聚类', fontproperties=font)
#ax2.set_title('Image after clustering')
axis('off')
imshow(codeim21)

subplot(236)
title(u'100×100窗格聚类', fontproperties=font)
#ax2.set_title('Image after clustering')
axis('off')
imshow(codeim22)

show()