
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d import axes3d


# 添加中文字体支持
from matplotlib.font_manager import FontProperties
font = FontProperties(fname="/System/Library/Fonts/PingFang.ttc", size=14)

fig = plt.figure()
ax = fig.gca(projection="3d")
# 生成三维样本点
X,Y,Z = axes3d.get_test_data(0.25)
# 在三维中绘制点 
ax.plot(X.flatten(),Y.flatten(),Z.flatten(),'o')

plt.title(u'mplot3d样本点绘制图', fontproperties=font)

plt.show()

# # 绘制三维点
# from mpl_toolkits.mplot3d import axes3d
with open('load_vggdata.py','r') as f:
    exec(f.read())

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot(points3D[0],points3D[1],points3D[2],'k.')

title(u'Merton样本数据绘制图', fontproperties=font)

plt.show()
