import stereo
import scipy.misc 
from PIL import Image
from pylab import *
#Fig5-10
# im_l = array(Image.open('../data/cones/im0.ppm').convert('L'),'f') 
# im_r = array(Image.open('../data/cones/im1.ppm').convert('L'),'f')
# #Fig5-11
im_l = array(Image.open('../data/cones/im0.ppm').convert('L'),'f') 
im_r = array(Image.open('../data/cones/im1.ppm').convert('L'),'f')

# 开始偏移，并设置步长 
steps = 12
start = 4 

# ncc 的宽度 
wid = 9

res1 = stereo.plane_sweep_ncc(im_l,im_r,start,steps,wid)

 
scipy.misc.imsave('../images/ch05/depth_ncc.png',res1)


res2 = stereo.plane_sweep_gauss(im_l,im_r,start,steps,wid)

 
scipy.misc.imsave('../images/ch05/depth_gauss.png',res2)

# 添加中文字体支持
from matplotlib.font_manager import FontProperties
font = FontProperties(fname="/System/Library/Fonts/PingFang.ttc", size=14)
 
figure()

subplot(221)
imshow(im_l)
title(u'im_l', fontproperties=font)
axis('off')

subplot(222)
imshow(im_r)
title(u'im_r', fontproperties=font)
axis('off')

subplot(223)
imshow(res1)
title(u'ncc', fontproperties=font)
axis('off')

subplot(224)
imshow(res2)
title(u'gauss', fontproperties=font)
axis('off')

show()
