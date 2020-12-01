import camera
from pylab import *
import numpy as np

# 载入点
points = loadtxt('../data/model_house/3D/house.p3d').T
points = vstack((points,ones(points.shape[1])))

# 设置照相机参数
P = hstack((eye(3),array([[0],[0],[-10]]))) 
cam = camera.Camera(P)
x = cam.project(points)

# 绘制投影
figure() 
plot(x[0],x[1],'k.') 
show()


# 创建变换
r = 0.05*np.random.rand(3)
rot = camera.rotation_matrix(r)
# 旋转矩阵和投影 figure()
for t in range(20):
	cam.P = dot(cam.P,rot) 
	x = cam.project(points) 
	plot(x[0],x[1],'k.')
	
	show()
