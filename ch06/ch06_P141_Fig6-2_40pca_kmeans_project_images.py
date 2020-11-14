import imtools
import pickle
from PIL import Image
from scipy.cluster.vq import *
from pylab import *

# 获取 selected-fontimages 文件下图像文件名，并保存在列表中 
imlist = imtools.get_imlist('../data/fontimages/a_thumbs/')
imnbr = len(imlist)

# 载入模型文件
# with open('a_pca_modes.pkl','rb') as f:
with open('../data/fontimages/font_pca_modes.pkl','rb') as f:
	immean = pickle.load(f) 
	V = pickle.load(f)

# 创建矩阵，存储所有拉成一组形式后的图像
immatrix = array([array(Image.open(im)).flatten() for im in imlist],'f')

# 投影到前 40 个主成分上 
immean = immean.flatten()
projected = array([dot(V[:40],immatrix[i]-immean) for i in range(imnbr)])

# 进行 k-means 聚类
projected = whiten(projected) 
centroids,distortion = kmeans(projected,4)

code,distance = vq(projected,centroids)

# 绘制聚类簇
for k in range(4):
	ind = where(code==k)[0]
	figure()
	gray()
	for i in range(minimum(len(ind),40)):
		subplot(4,10,i+1) 
		imshow(immatrix[ind[i]].reshape((25,25))) 
		axis('off')
show()