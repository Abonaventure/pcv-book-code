import camera
from pylab import *
from PIL import Image
import numpy as np
# 载入一些图像
im1 = array(Image.open('../data/MertonCollegeI/images/001.jpg')) 
im2 = array(Image.open('../data/MertonCollegeI/images/002.jpg'))
# 载入每个视图的二维点到列表中
points2D = [loadtxt('../data/MertonCollegeI/2D/00'+str(i+1)+'.corners').T for i in range(3)]
# 载入三维点
points3D = loadtxt('../data/MertonCollegeI/3D/p3d').T
# 载入对应
corr = genfromtxt('../data/MertonCollegeI/2D/nview-corners',dtype='int',missing_values='*')
# 载入照相机矩阵到 Camera 对象列表中
P = [camera.Camera(loadtxt('../data/MertonCollegeI/2D/00'+str(i+1)+'.P')) for i in range(3)]