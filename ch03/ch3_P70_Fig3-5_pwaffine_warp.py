 # -*- coding: utf-8 -*-
from pylab import *
from PIL import Image
# from PCV.geometry import warp
import warp

"""
This is the piecewise affine warp example from Section 3.2, Figure 3-5.
"""

# open image to warp
fromim = array(Image.open('../data/sunset_tree.jpg')) #960 × 1280，160 × 160
x, y = meshgrid(range(5), range(6))

x = (fromim.shape[1]/4) * x.flatten()
y = (fromim.shape[0]/5) * y.flatten()

# triangulate
tri = warp.triangulate_points(x, y)

# open image and destination points
im = array(Image.open('../data/turningtorso1.jpg'))#2592 × 3888，240 × 240

figure()
subplot(1, 3, 1)
axis('off')
imshow(im)

tp = loadtxt('../data/turningtorso1_points.txt', 'int')  # destination points
# tp = array([[76,158,158,76],[76,76,158,158],[1,1,1,1]])

# convert points to hom. coordinates (make sure they are of type int)
fp = array(vstack((y, x, ones((1, len(x))))), 'int')
tp = array(vstack((tp[:, 1], tp[:, 0], ones((1, len(tp))))), 'int')

# warp triangles
im = warp.pw_affine(fromim, im, fp, tp, tri)

# plot
subplot(1, 3, 2)
axis('off')
imshow(im)
subplot(1, 3, 3)
axis('off')
imshow(im)
warp.plot_mesh(tp[1], tp[0], tri)

show()