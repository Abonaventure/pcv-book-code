import camera
from numpy import *

K = array([[1000,0,500],[0,1000,300],[0,0,1]]) 
tmp = camera.rotation_matrix([0,0,1])[:3,:3] 
Rt = hstack((tmp,array([[50],[40],[30]])))
cam = camera.Camera(dot(K,Rt))

print(K,Rt)
print(cam.factor())
#计算出两组数据不一样，可能是由于rq计算结果具有二义性导致