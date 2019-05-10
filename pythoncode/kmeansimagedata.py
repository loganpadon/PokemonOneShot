# import the necessary packages
from sklearn.cluster import KMeans
import math
import skimage
import matplotlib.pyplot as plt
import argparse
import numpy as np
import cv2
from os import listdir

def get_colors(centroids):
	# converts centroids into RGB integer colors
	output=""
 
	# loop over the color of each cluster
	for color in centroids:
		c=skimage.color.lab2rgb([[color]])*255
		rgb=[]
		for i in range(3):
			rgb.append(round(c[0][0][i]))
		output=output+"\t"+str(rgb)
	return output

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--directory", required = True, help = "Path to the images")
ap.add_argument("-c", "--clusters", required = True, type = int,
	help = "# of clusters")
args = vars(ap.parse_args())
dir=listdir(args["directory"])
fileOut="testparams-"+str(args["clusters"])+".txt"
fp=open(fileOut,"w")
for f in dir:
	imageLoc=args["directory"]+"/"+f
	print f
	image = cv2.imread(imageLoc)
	image2 = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	image = skimage.color.rgb2lab(image2)
 
	# show our image
	#plt.figure()
	#plt.axis("off")
	#plt.imshow(image2)

	# reshape the image to be a list of pixels
	imagedata = image.reshape((image.shape[0] * image.shape[1], 3))
	# cluster the pixel intensities
	clt = KMeans(n_clusters = args["clusters"])
	clt.fit(imagedata)
	data = get_colors(clt.cluster_centers_)
	fp.write(f+data+"\n")
	print data


