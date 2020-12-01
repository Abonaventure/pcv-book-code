from pylab import *
from numpy import *
from PIL import Image

# If you have PCV installed, these imports should work
# from PCV.geometry import homography, warp
# from PCV.localdescriptors import sift

import homography, warp
import sift

"""
This is the panorama example from section 3.3.
"""

# set paths to data folder
featname = ['../data/Univ'+str(i+1)+'.sift' for i in range(5)] 
imname = ['../data/Univ'+str(i+1)+'.jpg' for i in range(5)]

# extract features and match
l = {}
d = {}
for i in range(5): 
    sift.process_image(imname[i],featname[i])
    l[i],d[i] = sift.read_features_from_file(featname[i])

matches = {}
for i in range(4):
    matches[i] = sift.match(d[i+1],d[i])

# print(matches)

# visualize the matches (Figure 3-11 in the book)
figure()
gray()
for i in range(4):
    subplot(2,2,i+1)
    im1 = array(Image.open(imname[i]))
    im2 = array(Image.open(imname[i+1]))
    sift.plot_matches(im2,im1,l[i+1],l[i],matches[i],show_below=True)

show()


