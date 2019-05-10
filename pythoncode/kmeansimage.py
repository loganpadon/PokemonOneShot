# import the necessary packages
from sklearn.cluster import KMeans
import skimage
import matplotlib.pyplot as plt
import argparse
import cv2

def mean_image(image,clt):
	image2=image
	for x in range(len(image2)):
		classes=clt.predict(image2[x])
		for y in range(len(classes)):
			image2[x,y]=clt.cluster_centers_[classes[y]]
	image2=skimage.color.lab2rgb(image2)
	
	return image2
def plot_colors(hist, centroids):
	# initialize the bar chart representing the relative frequency
	# of each of the colors
	bar = np.zeros((50, 300, 3), dtype = "uint8")
	startX = 0
 
	# loop over the percentage of each cluster and the color of
	# each cluster
	for (percent, color) in zip(hist, centroids):
		print color
		c = skimage.color.lab2rgb([[color]])
		print c*255
		# plot the relative percentage of each cluster
		endX = startX + (percent * 300)
		cv2.rectangle(bar, (int(startX), 0), (int(endX), 50),
			c[0][0]*255, -1)
		startX = endX
	
	# return the bar chart
	return bar
# import the necessary packages
import numpy as np
import cv2
 
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
ap.add_argument("-i", "--image", required = True, help = "Path to the image")
ap.add_argument("-c", "--clusters", required = True, type = int,
	help = "# of clusters")
args = vars(ap.parse_args())
 
# load the image and convert it from BGR to RGB so that
# we can dispaly it with matplotlib
image = cv2.imread(args["image"])
image2 = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = skimage.color.rgb2lab(image2)
 
# show our image
plt.figure()
plt.axis("off")
plt.imshow(image2)

# reshape the image to be a list of pixels
imagedata = image.reshape((image.shape[0] * image.shape[1], 3))
# cluster the pixel intensities
clt = KMeans(n_clusters = args["clusters"])
clt.fit(imagedata)
hist = centroid_histogram(clt)
bar = plot_colors(hist, clt.cluster_centers_)
# show our color bar
plt.figure()
plt.axis("off")
plt.imshow(bar)
imagek=mean_image(image,clt)
plt.figure()
plt.axis("off")
plt.imshow(imagek)

plt.show()

