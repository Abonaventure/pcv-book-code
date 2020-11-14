import os
# import Image
# from PCV.clustering import hcluster
from PIL import Image
import hcluster
# create a list of images
from matplotlib.pyplot import *
from numpy import *

# path = 'D:\\GitHub\\PCV-translation-to-Chinese\\data\\sunsets\\flickr-sunsets-small\\'
path = '../data/flickr-sunsets-small/'
imlist = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.jpg')]
# extract feature vector (8 bins per color channel)
features = zeros([len(imlist), 512])
for i, f in enumerate(imlist):
    im = array(Image.open(f))
    # multi-dimensional histogram
    h, edges = histogramdd(im.reshape(-1, 3), 8, normed=True, range=[(0, 255), (0, 255), (0, 255)])
    features[i] = h.flatten()
tree = hcluster.hcluster(features)

# visualize clusters with some (arbitrary) threshold
clusters = tree.extract_clusters(0.23 * tree.distance)
# plot images for clusters with more than 3 elements
# figure()
for c in clusters:
    elements = c.get_cluster_elements()
    nbr_elements = len(elements)
    if nbr_elements > 3:
        figure()
        subplot(1,2,2)
        for p in range(minimum(nbr_elements,20)):
            subplot(4, 5, p + 1)
            im = array(Image.open(imlist[elements[p]]))
            imshow(im)
            axis('off')
        # r = 5 - minimum(nbr_elements,20)%5
        # if r ==5 :
        #     for p in range(minimum(nbr_elements,20)):
        #         subplot(4, 5, p + 1)
        #         im = array(Image.open(imlist[elements[p]]))
        #         imshow(im)
        #         axis('off')
        # else:
        #     for p in range(minimum(nbr_elements,20)):
        #         subplot(4, 5, p + 1)
        #         im = array(Image.open(imlist[elements[p]]))
        #         imshow(im)
        #         axis('off')

        #     for i in range(r):
        #         subplot(4,5, 6-r+i)
        #         # imshow()
        #         axis('off')

show()