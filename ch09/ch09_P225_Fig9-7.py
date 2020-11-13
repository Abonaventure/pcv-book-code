import rof
from pylab import *
from PIL import Image
# import scipy.misc
import imageio
from skimage import *



im1 = array(Image.open('../data/flower32_t0.png').convert("L"))
im2 = array(Image.open('../data/ceramic-houses_t0.png').convert("L"))
U1, T1 = rof.denoise(im1, im1, tolerance=0.001)
U2, T2 = rof.denoise(im2, im2, tolerance=0.001)

t1 = 0.8  # flower32_t0 threshold
t2 = 0.4  # ceramic-houses_t0 threshold
seg_im1 = img_as_uint(U1 < t1*U1.max())
seg_im2 = img_as_uint(U2 < t2*U2.max())

fig = figure()
gray()
subplot(231)
axis('off')
imshow(im1)

subplot(232)
axis('off')
imshow(U1)

subplot(233)
axis('off')
imshow(seg_im1)

subplot(234)
axis('off')
imshow(im2)

subplot(235)
axis('off')
imshow(U2)

subplot(236)
axis('off')
imshow(seg_im2)

show()

# scipy.misc.imsave('../images/ch09/flower32_t0_result.pdf', seg_im)
imageio.imsave('../images/ch09/flower32_t0_result.pdf', seg_im1)
imageio.imsave('../images/ch09/ceramic-houses_t0_result.pdf', seg_im2)

# fig.savefig('../images/ch09/flower32_t0_result.pdf', seg_im1)
# fig.savefig('../images/ch09/ceramic-houses_t0_result.pdf', seg_im2)
