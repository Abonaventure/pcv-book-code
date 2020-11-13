from pylab import *
from scipy.cluster.vq import * 

def ncut_graph_matrix(im,sigma_d=1e2,sigma_g=1e-2):
# """ 创建用于归一化割的矩阵，其中 sigma_d 和 sigma_g 是像素距离和像素相似性的权重参数 """ 

	m,n = im.shape[:2]
	N = m*n

	# 归一化，并创建 RGB 或灰度特征向量 
	if len(im.shape)==3:
		for i in range(3):
			im[:,:,i] = im[:,:,i] / im[:,:,i].max()
		vim = im.reshape((-1,3)) 
	else:
		im = im / im.max() 
		vim = im.flatten()

	# x,y 坐标用于距离计算
	xx,yy = meshgrid(range(n),range(m)) 
	x,y = xx.flatten(),yy.flatten()

	# 创建边线权重矩阵
	W = zeros((N,N),'f') 
	for i in range(N):
		for j in range(i,N):
			d = (x[i]-x[j])**2 + (y[i]-y[j])**2
			W[i,j] = W[j,i] = exp(-1.0*sum((vim[i]-vim[j])**2)/sigma_g) * exp(-d/sigma_d)
	return W


def cluster(S,k,ndim):
	""" 从相似性矩阵进行谱聚类 """
	# 检查对称性
	if sum(abs(S-S.T)) > 1e-10:
		print('not symmetric')

	# 创建拉普拉斯矩阵
	rowsum = sum(abs(S),axis=0)
	D = diag(1 / sqrt(rowsum + 1e-6)) 
	L = dot(D,dot(S,D))

	# 计算 L 的特征向量 
	U,sigma,V = linalg.svd(L)

	# 从前 ndim 个特征向量创建特征向量 
	# 堆叠特征向量作为矩阵的列 
	features = array(V[:ndim]).T

	# K-means 聚类
	features = whiten(features) 
	centroids,distortion = kmeans(features,k) 
	code,distance = vq(features,centroids)

	return code,V
