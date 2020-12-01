
import sfm, camera

# execfile('load_vggdata.py')
# python3 删去了 execfile()，代替方法如下：
with open('load_vggdata.py','r') as f:
    exec(f.read())
corr = corr[:,0] # 视图 1
ndx3D = where(corr>=0)[0] # 丢失的数值为 -1 
ndx2D = corr[ndx3D]

# 选取可见点，并用齐次坐标表示
x = points2D[0][:,ndx2D] # 视图 1
x = vstack( (x,ones(x.shape[1])) ) 
X = points3D[:,ndx3D]
X = vstack( (X,ones(X.shape[1])) )

# 估计P
Pest = camera.Camera(sfm.compute_P(x,X))

# 比较!
print(Pest.P / Pest.P[2,3])
print(P[0].P / P[0].P[2,3]) 

xest = Pest.project(X)

# 绘制图像
figure()
imshow(im1) 
plot(x[0],x[1],'bo') 
plot(xest[0],xest[1],'r.') 
axis('off')

show()
