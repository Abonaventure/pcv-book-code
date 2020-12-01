import homography
import sfm
import sift
from pylab import *
from PIL import Image
  
# 标定矩阵
K = array([[2394,0,932],[0,2398,628],[0,0,1]])
# 载入图像，并计算特征
im1 = array(Image.open('../data/alcatraz1.jpg')) 
sift.process_image('../data/alcatraz1.jpg','im1.sift') 
l1,d1 = sift.read_features_from_file('im1.sift')
im2 = array(Image.open('../data/alcatraz2.jpg')) 
sift.process_image('../data/alcatraz2.jpg','im2.sift') 
l2,d2 = sift.read_features_from_file('im2.sift')
# 匹配特征
matches = sift.match_twosided(d1,d2) 
ndx = matches.nonzero()[0]
# 使用齐次坐标表示，并使用 inv(K) 归一化 
x1 = homography.make_homog(l1[ndx,:2].T) 
ndx2 = [int(matches[i]) for i in ndx]
x2 = homography.make_homog(l2[ndx2,:2].T)
x1n = dot(inv(K),x1) 
x2n = dot(inv(K),x2)
# 使用 RANSAC 方法估计 E
model = sfm.RansacModel()
E,inliers = sfm.F_from_ransac(x1n,x2n,model)
# 计算照相机矩阵(P2 是 4 个解的列表)
P1 = array([[1,0,0,0],[0,1,0,0],[0,0,1,0]]) 
P2 = sfm.compute_P_from_essential(E)

# 选取点在照相机前的解 ind = 0
maxres = 0
for i in range(4):
# 三角剖分正确点，并计算每个照相机的深度
	X = sfm.triangulate(x1n[:,inliers],x2n[:,inliers],P1,P2[i]) 
	d1 = dot(P1,X)[2]
	d2 = dot(P2[i],X)[2]
	if sum(d1>0)+sum(d2>0) > maxres:
		maxres = sum(d1>0)+sum(d2>0) 
		ind = i
		infront = (d1>0) & (d2>0)
# 三角剖分正确点，并移除不在所有照相机前面的点
X = sfm.triangulate(x1n[:,inliers],x2n[:,inliers],P1,P2[ind]) 
X = X[:,infront]

# 绘制三维图像
from mpl_toolkits.mplot3d import axes3d

fig = figure()

subplot(223)
ax = fig.gca(projection='3d') 
ax.plot(-X[0],X[1],X[2],'k.') 
axis('off')

subplot(224)
ax = fig.gca(projection='3d') 
ax.plot(-X[0],X[1],X[2],'k.') 
axis('off')

# 绘制X的投影
import camera

# 绘制三维点
cam1 = camera.Camera(P1)
cam2 = camera.Camera(P2[ind]) 
x1p = cam1.project(X)
x2p = cam2.project(X)

# 反K归一化
x1p = dot(K,x1p) 
x2p = dot(K,x2p)

 
subplot(221)
imshow(im1)
gray() 
plot(x1p[0],x1p[1],'o') 
plot(x1[0],x1[1],'r.') 
axis('off')

 
subplot(222)
imshow(im2)
gray() 
plot(x2p[0],x2p[1],'o') 
plot(x2[0],x2[1],'r.') 
axis('off')

show()
