# from PCV.tools import ncut
# from scipy.misc import imresize
import ncut
from pylab import *
from PIL import Image

im = array(Image.open('../data/C-uniform03.ppm'))
m, n = im.shape[:2]
print(n,m)
# resize image to (wid,wid)
wid = 50
# rim = imresize(im, (wid, wid), interp='bilinear')
rim = np.array(Image.fromarray(im).resize((wid,wid),Image.BILINEAR))
rim = array(rim, 'f')
# create normalized cut matrix
A = ncut.ncut_graph_matrix(rim, sigma_d=1, sigma_g=1e-2)
# cluster
code, V = ncut.cluster(A, k=3, ndim=3)
print(array(V).shape)
print("ncut finish")

# 变换到原来的图像大小
# codeim = imresize(code.reshape(wid,wid),(m,n),interp='nearest')
codeim = array(Image.fromarray(code.reshape(wid,wid)).resize((n,m),Image.NEAREST))
# imshow(imresize(V[i].reshape(wid,wid),(m,n),interp=’bilinear’))
# v = zeros((m,n,4),int)
v = zeros((4,m,n),int)
for i in range(4):
	v[i] = array(Image.fromarray(V[i].reshape(wid,wid)).resize((n,m),Image.BILINEAR))

# 绘制分割结果 
fig = figure()
gray()
subplot(242)
axis('off')
imshow(im)

subplot(243)
axis('off')
imshow(codeim)

for i in range(4):
	subplot(2,4,i+5)
	axis('off')
	imshow(v[i])

show()


