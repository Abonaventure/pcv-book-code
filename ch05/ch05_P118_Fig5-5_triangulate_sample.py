import sfm

# execfile('load_vggdata.py')
# python3 删去了 execfile()，代替方法如下：
with open('load_vggdata.py','r') as f:
    exec(f.read())
# 前两个视图中点的索引
ndx = (corr[:,0]>=0) & (corr[:,1]>=0)

# 获取坐标，并用齐次坐标表示
x1 = points2D[0][:,corr[ndx,0]]
x1 = vstack( (x1,ones(x1.shape[1])) ) 
x2 = points2D[1][:,corr[ndx,1]]
x2 = vstack( (x2,ones(x2.shape[1])) )

Xtrue = points3D[:,ndx]
Xtrue = vstack( (Xtrue,ones(Xtrue.shape[1])) )

# 检查前三个点
Xest = sfm.triangulate(x1,x2,P[0].P,P[1].P) 
print(Xest[:,:3])
print(Xtrue[:,:3])

# 绘制图像
from mpl_toolkits.mplot3d import axes3d 
fig = figure()
ax = fig.gca(projection='3d') 
ax.plot(Xest[0],Xest[1],Xest[2],'ko') 
ax.plot(Xtrue[0],Xtrue[1],Xtrue[2],'r.') 
# axis('equal')
axis()

show()