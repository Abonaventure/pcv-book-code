
from OpenGL.GL import * 
from OpenGL.GLU import * 
from OpenGL.GLUT import * 
import pygame, pygame.image 
from pygame.locals import * 
from pylab import *
import pickle
import sift
import homography
import camera

def cube_points(c, wid):
    """ Creates a list of points for plotting
        a cube with plot. (the first 5 points are
        the bottom square, some sides repeated). """
    p = []
    # bottom
    p.append([c[0]-wid, c[1]-wid, c[2]-wid])
    p.append([c[0]-wid, c[1]+wid, c[2]-wid])
    p.append([c[0]+wid, c[1]+wid, c[2]-wid])
    p.append([c[0]+wid, c[1]-wid, c[2]-wid])
    p.append([c[0]-wid, c[1]-wid, c[2]-wid]) #same as first to close plot
    
    # top
    p.append([c[0]-wid, c[1]-wid, c[2]+wid])
    p.append([c[0]-wid, c[1]+wid, c[2]+wid])
    p.append([c[0]+wid, c[1]+wid, c[2]+wid])
    p.append([c[0]+wid, c[1]-wid, c[2]+wid])
    p.append([c[0]-wid, c[1]-wid, c[2]+wid]) #same as first to close plot
    
    # vertical sides
    p.append([c[0]-wid, c[1]-wid, c[2]+wid])
    p.append([c[0]-wid, c[1]+wid, c[2]+wid])
    p.append([c[0]-wid, c[1]+wid, c[2]-wid])
    p.append([c[0]+wid, c[1]+wid, c[2]-wid])
    p.append([c[0]+wid, c[1]+wid, c[2]+wid])
    p.append([c[0]+wid, c[1]-wid, c[2]+wid])
    p.append([c[0]+wid, c[1]-wid, c[2]-wid])
    
    return array(p).T
    
def my_calibration(sz):
    row, col = sz
    fx = 2555*col/2592
    fy = 2586*row/1936
    K = diag([fx, fy, 1])
    K[0, 2] = 0.5*col
    K[1, 2] = 0.5*row
    return K

def set_projection_from_camera(K): 
	""" 从照相机标定矩阵中获得视图 """
	glMatrixMode(GL_PROJECTION)
	glLoadIdentity()
	fx = K[0,0]
	fy = K[1,1]
	fovy = 2*arctan(0.5*height/fy)*180/pi
	aspect = (width*fy)/(height*fx)
	# 定义近的和远的剪裁平面 
	near = 0.1
	far = 100.0
	# 设定透视 
	gluPerspective(fovy,aspect,near,far) 
	glViewport(0,0,width,height)

def set_modelview_from_camera(Rt):
	""" 从照相机姿态中获得模拟视图矩阵 """
	glMatrixMode(GL_MODELVIEW)
	glLoadIdentity()
	# 围绕x 轴将茶壶旋转 90 度，使z 轴向上 
	Rx = array([[1,0,0],[0,0,-1],[0,1,0]])
	# 获得旋转的最佳逼近
	R = Rt[:,:3]
	U,S,V = linalg.svd(R)
	R = dot(U,V)
	R[0,:] = -R[0,:] # 改变 x 轴的符号
	# 获得平移量 
	t = Rt[:,3]
	# 获得 4×4 的模拟视图矩阵 
	M = eye(4)
	M[:3,:3] = dot(R,Rx) 
	M[:3,3] = t
	# 转置并压平以获取列序数值 
	M = M.T
	m = M.flatten()
	# 将模拟视图矩阵替换为新的矩阵 
	glLoadMatrixf(m)

def draw_background(imname): 
	""" 使用四边形绘制背景图像 """
	# 载入背景图像(应该是 .bmp 格式)，转换为 OpenGL 纹理 
	bg_image = pygame.image.load(imname).convert() 
	bg_data = pygame.image.tostring(bg_image,"RGBX",1)

	glMatrixMode(GL_MODELVIEW)
	glLoadIdentity()
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

	# 绑定纹理
	glEnable(GL_TEXTURE_2D)
	glBindTexture(GL_TEXTURE_2D,glGenTextures(1)) 
	glTexImage2D(GL_TEXTURE_2D,0,GL_RGBA,width,height,0,GL_RGBA,GL_UNSIGNED_BYTE,bg_data)
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)

    # 创建四方形填充整个窗口
	glBegin(GL_QUADS)
	glTexCoord2f(0.0,0.0);glVertex3f(-1.0,-1.0,-1.0) 
	glTexCoord2f(1.0,0.0);glVertex3f( 1.0,-1.0,-1.0) 
	glTexCoord2f(1.0,1.0);glVertex3f( 1.0, 1.0,-1.0) 
	glTexCoord2f(0.0,1.0);glVertex3f(-1.0, 1.0,-1.0) 
	glEnd()
	
	# 清除纹理 
	glDeleteTextures(1)

def draw_teapot(size):
	""" 在原点处绘制红色茶壶 """
	glEnable(GL_LIGHTING)
	glEnable(GL_LIGHT0)
	glEnable(GL_DEPTH_TEST)
	glClear(GL_DEPTH_BUFFER_BIT)
	# 绘制红色茶壶
	glMaterialfv(GL_FRONT,GL_AMBIENT,[0,0,0,0])
	glMaterialfv(GL_FRONT,GL_DIFFUSE,[0.5,0.0,0.0,0.0])
	glMaterialfv(GL_FRONT,GL_SPECULAR,[0.7,0.6,0.6,0.0])
	glMaterialf(GL_FRONT,GL_SHININESS,0.25*128.0)
	glutSolidTeapot(size)

def load_and_draw_model(filename):
	""" 使用 objloader.py，从 .obj 文件中装载模型 假设路径文件夹中存在同名的 .mtl 材料设置文件 """ 
	glEnable(GL_LIGHTING)
	glEnable(GL_LIGHT0)
	glEnable(GL_DEPTH_TEST) 
	glClear(GL_DEPTH_BUFFER_BIT)
	# 设置模型颜色
	glMaterialfv(GL_FRONT,GL_AMBIENT,[0,0,0,0]) 
	glMaterialfv(GL_FRONT,GL_DIFFUSE,[0.5,0.75,1.0,0.0]) 
	glMaterialf(GL_FRONT,GL_SHININESS,0.25*128.0)
	# 从文件中载入
	import objloader
	obj = objloader.OBJ(filename,swapyz=True) 
	glCallList(obj.gl_list)

# # compute features
# sift.process_image('../data/book_frontal.JPG', 'im0.sift')
# l0, d0 = sift.read_features_from_file('im0.sift')

# sift.process_image('../data/book_perspective.JPG', 'im1.sift')
# l1, d1 = sift.read_features_from_file('im1.sift')


# # match features and estimate homography
# matches = sift.match_twosided(d0, d1)
# ndx = matches.nonzero()[0]
# fp = homography.make_homog(l0[ndx, :2].T)
# ndx2 = [int(matches[i]) for i in ndx]
# tp = homography.make_homog(l1[ndx2, :2].T)

# model = homography.RansacModel()
# H, inliers = homography.H_from_ransac(fp, tp, model)

# # camera calibration
# K = my_calibration((747, 1000))

# # 3D points at plane z=0 with sides of length 0.2
# box = cube_points([0, 0, 0.1], 0.1)

# # project bottom square in first image
# cam1 = camera.Camera(hstack((K, dot(K, array([[0], [0], [-1]])))))
# # first points are the bottom square
# box_cam1 = cam1.project(homography.make_homog(box[:, :5]))


# # use H to transfer points to the second image
# box_trans = homography.normalize(dot(H,box_cam1))

# # compute second camera matrix from cam1 and H
# cam2 = camera.Camera(dot(H, cam1.P))
# A = dot(linalg.inv(K), cam2.P[:, :3])
# A = array([A[:, 0], A[:, 1], cross(A[:, 0], A[:, 1])]).T
# cam2.P[:, :3] = dot(K, A)

# # project with the second camera
# box_cam2 = cam2.project(homography.make_homog(box))

# # 测试:将点投影在 z=0 上，应该能够得到相同的点
# point = array([1,1,0,1]).T
# print(homography.normalize(dot(dot(H,cam1.P),point)) ) 
# print(cam2.project(point)) 

# # Rt = dot(linalg.inv(K), cam2.P)
# import pickle
# with open('../data/ar_camera.pkl','wb') as f: 
#     pickle.dump(K,f,0) 
#     pickle.dump(dot(linalg.inv(K),cam2.P),f)

width,height = 1000,747
def setup():
	""" 设置窗口和 pygame 环境 """
	pygame.init() 
	pygame.display.set_mode((width,height),OPENGL | DOUBLEBUF) 
	pygame.display.set_caption('OpenGL AR demo')

# 载入照相机数据
with open('../data/ar_camera.pkl','rb') as f:
	K = pickle.load(f) 
	Rt = pickle.load(f)

setup() 
draw_background('../data/book_perspective.bmp') 
set_projection_from_camera(K) 
set_modelview_from_camera(Rt) 
# draw_teapot(0.1)
load_and_draw_model('../data/toyplane.obj')

pygame.display.flip()
while True:
  for event in pygame.event.get():
    if event.type == pygame.QUIT:
       sys.exit()
       pygame.quit()
# pygame.display.flip()


