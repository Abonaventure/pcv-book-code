# -*- coding: utf-8 -*-
from pylab import *
from PIL import Image
# from PCV.localdescriptors import sift
# from PCV.tools import imtools
import sift
import imtools
import pydot
import os

""" This is the example graph illustration of matching images from Figure 2-10.
To download the images, see ch2_download_panoramio.py."""
# download_path = '/Users/Abner/Documents/Codes/python_work/pcv/pcv-book-code/data/panoimages/'  # set this to the path where you downloaded the panoramio images
# path = '/Users/Abner/Documents/Codes/python_work/pcv/pcv-book-code/ch02/'  # path to save thumbnails (pydot needs the full system path)

parent_path = os.path.abspath(os.path.join(os.path.dirname("__file__"),os.path.pardir))
download_path = parent_path+'/data/panoimages/'  # set this to the path where you downloaded the panoramio images
path = parent_path+'/data/panoimages/thumbnails/'  # path to save thumbnails (pydot needs the full system path)
# list of downloaded filenames
imlist = imtools.get_imlist(download_path)
nbr_images = len(imlist)

# extract features
featlist = [imname[:-3] + 'sift' for imname in imlist]


matchscores = zeros((nbr_images,nbr_images))
for i in range(nbr_images):
    for j in range(i, nbr_images):  # only compute upper triangle
        # print('comparing ', imlist[i],imlist[j])
        # process and save features to file
        sift.process_image(imlist[i],featlist[i])
        l1, d1 = sift.read_features_from_file(featlist[i])
        sift.process_image(imlist[j],featlist[j])
        l2, d2 = sift.read_features_from_file(featlist[j])
        matches = sift.match_twosided(d1, d2)
        nbr_matches = sum(matches > 0)
        # print('number of matches = ', nbr_matches) 
        matchscores[i, j] = nbr_matches

# copy values
for i in range(nbr_images):
    for j in range(i + 1, nbr_images):  # no need to copy diagonal
        matchscores[j, i] = matchscores[i, j]

print("The match scores is: %d",matchscores)
#savetxt(("../data/panoimages/panoramio_matches.txt",matchscores)

threshold = 2  # min number of matches needed to create link

g = pydot.Dot(graph_type='graph')  # don't want the default directed graph

for i in range(nbr_images):
    for j in range(i + 1, nbr_images):
        if matchscores[i, j] > threshold:
            # first image in pair
            im = Image.open(imlist[i])
            im.thumbnail((100, 100))
            filename = path+str(i) + '.png'
            im.save(filename)  # need temporary files of the right size
            g.add_node(pydot.Node(str(i), fontcolor='transparent', \
                shape='rectangle', image=filename))

            # second image in pair
            im = Image.open(imlist[j])
            im.thumbnail((100, 100))
            filename =  path+str(j) + '.png'
            im.save(filename)  # need temporary files of the right size
            g.add_node(pydot.Node(str(j), fontcolor='transparent', \
                shape='rectangle', image=filename))

            g.add_edge(pydot.Edge(str(i), str(j)))
# g.write_png('whitehouse.png')
g.write('whitehouse.png',format='png')



from PIL import Image
from pylab import *
 
# 添加中文字体支持
from matplotlib.font_manager import FontProperties
font = FontProperties(fname="/System/Library/Fonts/PingFang.ttc", size=14)

figure()
im = array(Image.open('whitehouse.png'))
imshow(im)
title(u'MatchGraph', fontproperties=font)
axis('off')
show()