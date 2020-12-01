
import homography
import camera
import sift

def my_calibration(sz): 
	row,col = sz
	fx = 2555*col/2592
	fy = 2586*row/1936
	K = diag([fx,fy,1]) 
	K[0,2] = 0.5*col 
	K[1,2] = 0.5*row 
	return K


def cube_points(c,wid):
	""" 创建用于绘制立方体的一个点列表(前 5 个点是底部的正方形，一些边重合了)""" 
	p = []
	# 底部
	p.append([c[0]-wid,c[1]-wid,c[2]-wid]) 
	p.append([c[0]-wid,c[1]+wid,c[2]-wid])

	p.append([c[0]+wid,c[1]+wid,c[2]-wid]) 
	p.append([c[0]+wid,c[1]-wid,c[2]-wid]) 
	p.append([c[0]-wid,c[1]-wid,c[2]-wid]) 
	# 为了绘制闭合图像，和第一个相同
	# 顶部
	p.append([c[0]-wid,c[1]-wid,c[2]+wid]) 
	p.append([c[0]-wid,c[1]+wid,c[2]+wid]) 
	p.append([c[0]+wid,c[1]+wid,c[2]+wid]) 
	p.append([c[0]+wid,c[1]-wid,c[2]+wid]) 
	p.append([c[0]-wid,c[1]-wid,c[2]+wid]) 
	# 为了绘制闭合图像，和第一个相同
	# 竖直边 
	p.append([c[0]-wid,c[1]-wid,c[2]+wid]) 
	p.append([c[0]-wid,c[1]+wid,c[2]+wid]) 
	p.append([c[0]-wid,c[1]+wid,c[2]-wid]) 
	p.append([c[0]+wid,c[1]+wid,c[2]-wid]) 
	p.append([c[0]+wid,c[1]+wid,c[2]+wid]) 
	p.append([c[0]+wid,c[1]-wid,c[2]+wid]) 
	p.append([c[0]+wid,c[1]-wid,c[2]-wid])
	return array(p).T

# 计算特征 
sift.process_image('../data/book_frontal.JPG','im0.sift') 
l0,d0 = sift.read_features_from_file('im0.sift')

sift.process_image('../data/book_perspective.JPG','im1.sift') 
l1,d1 = sift.read_features_from_file('im1.sift')

# 匹配特征，并计算单应性矩阵
matches = sift.match_twosided(d0,d1)
ndx = matches.nonzero()[0]
fp = homography.make_homog(l0[ndx,:2].T) 
ndx2 = [int(matches[i]) for i in ndx]
tp = homography.make_homog(l1[ndx2,:2].T)

model = homography.RansacModel()
H = homography.H_from_ransac(fp,tp,model)

im0 = array(Image.open('../data/book_frontal.JPG'))
im1 = array(Image.open('../data/book_perspective.JPG'))
# 底部正方形的二维投影
figure()
imshow(im0) 
plot(box_cam1[0,:],box_cam1[1,:],linewidth=3)
# 使用 H 对二维投影进行变换
figure()
imshow(im1) 
plot(box_trans[0,:],box_trans[1,:],linewidth=3)
# 三维立方体
figure()
imshow(im1) 
plot(box_cam2[0,:],box_cam2[1,:],linewidth=3)
show()


import pickle
with open('ar_camera.pkl','w') as f: 
	pickle.dump(K,f) 
	pickle.dump(dot(linalg.inv(K),cam2.P),f)