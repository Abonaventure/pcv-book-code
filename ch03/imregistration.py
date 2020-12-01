 # -*- coding: utf-8 -*-
from pylab import *
from xml.dom import minidom
from scipy import linalg,ndimage
from scipy.misc import imsave 
from PIL import Image
import os


def read_points_from_xml(xmlFileName): 
	""" 读取用于人脸对齐的控制点 """

	xmldoc = minidom.parse(xmlFileName)
	facelist = xmldoc.getElementsByTagName('face') 
	faces = {}
	for xmlFace in facelist:
		fileName = xmlFace.attributes['file'].value 
		xf = int(xmlFace.attributes['xf'].value)
		yf = int(xmlFace.attributes['yf'].value)
		xs = int(xmlFace.attributes['xs'].value)
		ys = int(xmlFace.attributes['ys'].value)
		xm = int(xmlFace.attributes['xm'].value)
		ym = int(xmlFace.attributes['ym'].value) 
		faces[fileName] = array([xf, yf, xs, ys, xm, ym])
	return faces


# from scipy import linalg
def compute_rigid_transform(refpoints,points):
	""" 计算用于将点对齐到参考点的旋转、尺度和平移量 """
	A = array([ [points[0], -points[1], 1, 0], 
				[points[1], points[0], 0, 1], 
				[points[2], -points[3], 1, 0], 
				[points[3], points[2], 0, 1],
				[points[4], -points[5], 1, 0], 
				[points[5], points[4], 0, 1]])

	y = array([ refpoints[0], 
				refpoints[1], 
				refpoints[2], 
				refpoints[3], 
				refpoints[4],
				refpoints[5]])

	# 计算最小化 ||Ax-y|| 的最小二乘解
	a,b,tx,ty = linalg.lstsq(A,y)[0]
	R = array([[a, -b], [b, a]]) # 包含尺度的旋转矩阵

	return R,tx,ty

# from PIL import Image
# from scipy import ndimage 
# from scipy.misc import imsave 
# import os
def rigid_alignment(faces,path,plotflag=False): 
	""" 严格对齐图像，并将其保存为新的图像
	path 决定对齐后图像保存的位置 
	设置 plotflag=True，以绘制图像 """

	# 将第一幅图像中的点作为参考点 
	# refpoints = faces.values()[0]  #TypeError: 'dict_values' object is not subscriptable
	refpoints = list(faces.values())[0]
	# 使用仿射变换扭曲每幅图像 
	for face in faces:
		points = faces[face]
	R,tx,ty = compute_rigid_transform(refpoints, points) 
	T = array([[R[1][1], R[1][0]], [R[0][1], R[0][0]]])

	im = array(Image.open(os.path.join(path,face))) 
	im2 = zeros(im.shape, 'uint8')

	# 对每个颜色通道进行扭曲
	for i in range(len(im.shape)):
		im2[:,:,i] = ndimage.affine_transform(im[:,:,i],linalg.inv(T),offset=[-ty,-tx])
	
	if plotflag: 
		imshow(im2) 
		show()

	# 裁剪边界，并保存对齐后的图像 
	h,w = im2.shape[:2]
	border = (w+h)//20

	# 裁剪边界
	# imsave(os.path.join(path, 'aligned/'+face),im2[border:h-border,border:w-border,:])
	imsave(os.path.join(path, 'aligned/'+face),im2[border:h-border,border:w-border,:])


