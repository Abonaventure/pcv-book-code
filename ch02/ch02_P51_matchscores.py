import sift
import imtools
from pylab import *

#获取图像列表
imlist = imtools.get_imlist('../data/panoimages/')
nbr_images = len(imlist)

print("imlist\n")
print(imlist)
nbr_images = len(imlist)
print(nbr_images)

# extract features
featlist = [imname[:-3] + 'sift' for imname in imlist]

print("featlist\n")
print(featlist)

matchscores = zeros((nbr_images,nbr_images)) 
for i in range(nbr_images):
	for j in range(i,nbr_images): # 仅仅计算上三角
		print('comparing ', imlist[i], imlist[j]) 
		# process and save features to file
		sift.process_image(imlist[i],featlist[i])
		l1,d1 = sift.read_features_from_file(featlist[i])
		sift.process_image(imlist[j],featlist[j])
		l2,d2 = sift.read_features_from_file(featlist[j])

		matches = sift.match_twosided(d1,d2)

		nbr_matches = sum(matches > 0)
		print('number of matches = ', nbr_matches) 
		matchscores[i,j] = nbr_matches

# 复制值
for i in range(nbr_images):
	for j in range(i+1,nbr_images): # 不需要复制对角线 
		matchscores[j,i] = matchscores[i,j]

print("matchscores:\n",matchscores)
