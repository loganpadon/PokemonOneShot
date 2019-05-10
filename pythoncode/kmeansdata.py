# import the necessary packages
from sklearn.cluster import KMeans
import math
import skimage
import matplotlib.pyplot as plt
import argparse
import numpy as np
import cv2

def mean_image(image,clt):
	image2=image
	for x in range(len(image2)):
		classes=clt.predict(image2[x])
		for y in range(len(classes)):
			image2[x,y]=clt.cluster_centers_[classes[y]]
	image2=skimage.color.lab2rgb(image2)
	
	return image2
def link_colors(hist, centroids):
	# initialize the bar chart representing the relative frequency
	# of each of the colors
	output=""
 
	# loop over the percentage of each cluster and the color of
	# each cluster
	for (percent, color) in zip(hist, centroids):
		c=skimage.color.lab2rgb([[color]])*255
		output=output+"\t"+str(c[0][0])+" "+str(percent)
	return output
def getNormalization(centroids):
	colors=[]
	for color in centroids:
		c=skimage.color.lab2rgb([[color]])*255
		colors.append(c[0][0])
	z=0
	for r in range(256):
		for g in range(256):
			for b in range(256):
				dist=3*256
				for color in colors:
					cdist=math.sqrt(math.pow(color[0]-r,2)+math.pow(color[1]-g,2)+math.pow(color[2]-b,2))
					if cdist<dist:
						dist=cdist
				if dist>0:
					z+=1/dist
	return z
					
 
def centroid_histogram(clt):
	# grab the number of different clusters and create a histogram
	# based on the number of pixels assigned to each cluster
	numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
	(hist, _) = np.histogram(clt.labels_, bins = numLabels)
 
	# normalize the histogram, such that it sums to one
	hist = hist.astype("float")
	hist /= hist.sum()
 
	# return the histogram
	return hist 
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
#ap.add_argument("-i", "--image", required = True, help = "Path #to the image")
ap.add_argument("-c", "--clusters", required = True, type = int,
	help = "# of clusters")
args = vars(ap.parse_args())
 
# load the image and convert it from BGR to RGB so that
# we can dispaly it with matplotlib
fileOut="params-"+str(args["clusters"])+".txt"
fp=open(fileOut,"w")
for x in range(720):
	imageLoc=str(x+10)+".png"
	print imageLoc
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
	hist = centroid_histogram(clt)
	data = link_colors(hist, clt.cluster_centers_)
	z=getNormalization(clt.cluster_centers_)
	print z
	fp.write(str(x+10)+"\t"+str(z)+data+"\n")
	print data


