
# execfile('load_vggdata.py')
# python3 删去了 execfile()，代替方法如下：
with open('load_vggdata.py','r') as f:
    exec(f.read())

X = vstack( (points3D,ones(points3D.shape[1])) ) 
x = P[0].project(X)

# 在视图1中绘制点
figure()
imshow(im1) 
plot(points2D[0][0],points2D[0][1],'*') 
axis('off')

figure()
imshow(im1) 
plot(x[0],x[1],'r.') 
axis('off')

show()


