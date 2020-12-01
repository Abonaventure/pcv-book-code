import sfm

# 添加中文字体支持
from matplotlib.font_manager import FontProperties
font = FontProperties(fname="/System/Library/Fonts/PingFang.ttc", size=14)

# execfile('load_vggdata.py')
# python3 删去了 execfile()，代替方法如下：
with open('load_vggdata.py','r') as f:
    exec(f.read())
# 在前两个视图中点的索引
ndx = (corr[:,0]>=0) & (corr[:,1]>=0)

# 获得坐标，并将其用齐次坐标表示
x1 = points2D[0][:,corr[ndx,0]]
x1 = vstack( (x1,ones(x1.shape[1])) ) 
x2 = points2D[1][:,corr[ndx,1]]
x2 = vstack( (x2,ones(x2.shape[1])) )

# 计算F
F = sfm.compute_fundamental(x1,x2)

# 计算极点
e = sfm.compute_epipole(F)

# 绘制图像 
figure() 

subplot(121)
imshow(im1)
# 分别绘制每个点，这样会绘制出和线同样的颜色 
for i in range(5):
	plot(x2[0,i],x2[1,i],'o') 
title(u'外极点', fontproperties=font)
axis('off')

subplot(122)
imshow(im2)
# 分别绘制每条线，这样会绘制出很漂亮的颜色 
for i in range(5):
	sfm.plot_epipolar_line(im2,F,x2[:,i],e,False) 
title(u'外极线', fontproperties=font)
axis('off')

show()
