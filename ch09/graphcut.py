from pygraph.classes.digraph import digraph
from pygraph.algorithms.minmax import maximum_flow
import bayes
# import numpy as np
# import matplotlib.pyplot as plt
from pylab import *

def build_bayes_graph(im,labels,sigma=1e2,kappa=2):
### """从像素四邻域建立一个图，前景和背景(前景用 1 标记，背景用 -1 标记，
### 其他的用 0 标记)由 labels 决定，并用朴素贝叶斯分类器建模 """ 

	m,n = im.shape[:2]

	# 每行是一个像素的 RGB 向量 
	vim = im.reshape((-1,3))

	# 前景和背景(RGB)
	foreground = im[labels==1].reshape((-1,3)) 
	background = im[labels==-1].reshape((-1,3)) 
	train_data = [foreground,background]

	# 训练朴素贝叶斯分类器
	bc = bayes.BayesClassifier() 
	bc.train(train_data)

	# 获取所有像素的概率 
	bc_lables,prob = bc.classify(vim) 
	prob_fg = prob[0]
	prob_bg = prob[1]

	# 用m*n+2 个节点创建图
	gr = digraph() 
	gr.add_nodes(range(m*n+2))
	source = m*n # 倒数第二个是源点 
	sink = m*n+1 # 最后一个节点是汇点

	# 归一化
	for i in range(vim.shape[0]):
		vim[i] = vim[i] / linalg.norm(vim[i])

	# 遍历所有的节点，并添加边 
	for i in range(m*n):
		# 从源点添加边
		gr.add_edge((source,i), wt=(prob_fg[i]/(prob_fg[i]+prob_bg[i])))
		# 向汇点添加边
		gr.add_edge((i,sink), wt=(prob_bg[i]/(prob_fg[i]+prob_bg[i])))

	# 向相邻节点添加边
	if i%n != 0: #左边存在
		edge_wt = kappa*np.exp(-1.0*np.sum((vim[i]-vim[i-1])**2)/sigma)
		gr.add_edge((i,i-1), wt=edge_wt) 
	if (i+1)%n != 0: # 如果右边存在
		edge_wt = kappa*np.exp(-1.0*np.sum((vim[i]-vim[i+1])**2)/sigma)
		gr.add_edge((i,i+1), wt=edge_wt) 
	if i//n != 0: #如果上方存在
		edge_wt = kappa*np.exp(-1.0*np.sum((vim[i]-vim[i-n])**2)/sigma)
		gr.add_edge((i,i-n), wt=edge_wt) 
	if i//n != m-1: # 如果下方存在
		edge_wt = kappa*np.exp(-1.0*np.sum((vim[i]-vim[i+n])**2)/sigma) 
		gr.add_edge((i,i+n), wt=edge_wt)

	return gr

def show_labeling(im,labels):
	###""" 显示图像的前景和背景区域。前景 labels=1, 背景 labels=-1，其他 labels = 0 """
	imshow(im)
	contour(labels,[-0.5,0.5]) 
	contourf(labels,[-1,-0.5],colors='b',alpha=0.25) 
	contourf(labels,[0.5,1],colors='r',alpha=0.25) 
	axis('off')

def cut_graph(gr,imsize):
	###""" 用最大流对图 gr 进行分割，并返回分割结果的二值标记 """
	m,n = imsize
	source = m*n # 倒数第二个节点是源点 
	sink = m*n+1 # 倒数第一个是汇点

	# 对图进行分割
	flows,cuts = maximum_flow(gr,source,sink)

	# 将图转为带有标记的图像
	res = zeros(m*n)
	for pos,label in list(cuts.items())[:-2]: # 不要添加源点 / 汇点
		res[pos] = label

	return res.reshape((m,n))
