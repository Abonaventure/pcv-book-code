from numpy import *
class BayesClassifier(object): 
	def __init__(self):
		""" 使用训练数据初始化分类器 """
		self.labels = [] # 类标签 
		self.mean = [] # 类均值 
		self.var = [] # 类方差 
		self.n = 0 #类别数

	def train(self,data,labels=None):
		""" 在数据 data(n×dim 的数组列表)上训练，标记 labels 是可选的，默认为 0...n-1 """
		if labels==None:
			labels = range(len(data))
		self.labels = labels 
		self.n = len(labels)

		for c in data: 
			self.mean.append(mean(c,axis=0)) 
			self.var.append(var(c,axis=0))

	

	def classify(self,points):
		""" 通过计算得出的每一类的概率对数据点进行分类，并返回最可能的标记 """

		# 计算每一类的概率
		est_prob = array([gauss(m,v,points) for m,v in zip(self.mean,self.var)])

		# 获取具有最高概率的索引，该索引会给出类标签
		ndx = est_prob.argmax(axis=0)
		est_labels = array([self.labels[n] for n in ndx])
		          
		return est_labels, est_prob
def gauss(m,v,x):
		""" 用独立均值m和方差v评估d维高斯分布 """
		if len(x.shape)==1: 
			n,d = 1,x.shape[0]
		else:
			n,d = x.shape
		# 协方差矩阵，减去均值 
		S = diag(1/v)
		x = x-m
		# 概率的乘积
		y = exp(-0.5*diag(dot(x,dot(S,x.T))))
		# 归一化并返回
		return y * (2*pi)**(-d/2.0) / ( sqrt(prod(v)) + 1e-6)
	