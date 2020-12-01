 # -*- coding: utf-8 -*-
# from PCV.geometry import warp, homography
import warp, homography
from PIL import  Image
from pylab import *
from scipy import ndimage

# example of affine warp of im1 onto im2

im1 = array(Image.open('../data/flickr_image/beatles.png').convert('L'))  #1483 × 957,300x300
im2 = array(Image.open('../data/flickr_image/billboard_for_rent.jpg').convert('L'))  #1475 × 2290,72x72
print(im1.shape[:2])
# set to points  
##tp在该程序中实际上是beatles图片要插在billboard图片上区域的四个顶点坐标(从左上角逆时针依次取点)，tp= （[x1,x2,x3,x4],[y1,y2,y3,y4],[1,1,1,1]])
# tp = array([[264,538,540,264],[40,36,605,605],[1,1,1,1]])
# tp = array([[675,826,826,677],[55,52,281,277],[1,1,1,1]])
tp = array([[580,1400,1400,580],[95,80,1365,1365],[1,1,1,1]])
print(tp)
im3 = warp.image_in_image(im1,im2,tp)
figure()
gray()
subplot(141)
axis('off')
imshow(im1)
subplot(142)
axis('off')
imshow(im2)
subplot(143)
axis('off')
imshow(im3)

tp = array([[1500,1850,1850,1500],[115,110,615,615],[1,1,1,1]])
# set from points to corners of im1
m,n = im1.shape[:2]
fp = array([[0,m,m,0],[0,0,n,n],[1,1,1,1]])
# first triangle
tp2 = tp[:,:3]
fp2 = fp[:,:3]
# compute H
H = homography.Haffine_from_points(tp2,fp2)
im1_t = ndimage.affine_transform(im1,H[:2,:2],
(H[0,2],H[1,2]),im2.shape[:2])
# alpha for triangle
alpha = warp.alpha_for_triangle(tp2,im2.shape[0],im2.shape[1])
im3 = (1-alpha)*im2 + alpha*im1_t
# second triangle
tp2 = tp[:,[0,2,3]]
fp2 = fp[:,[0,2,3]]
# compute H
H = homography.Haffine_from_points(tp2,fp2)
im1_t = ndimage.affine_transform(im1,H[:2,:2],
(H[0,2],H[1,2]),im2.shape[:2])
# alpha for triangle
alpha = warp.alpha_for_triangle(tp2,im2.shape[0],im2.shape[1])
im4 = (1-alpha)*im3 + alpha*im1_t
subplot(144)
imshow(im4)
axis('off')
show()