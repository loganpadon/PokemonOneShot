The python code contains various dependancies:
	scikit-learn
	scikit-image
	opencv
	numpy

kmeansdata.py- converts class images into class means data files
kmeansimagedata.py -takes the folder of test images and outputs a file containing their means
kmeansimage.py- displays the k-means image of the input image
naiveMeanClassify.py takes input class mean file and test image means file and outputs
	 accuracy of a naive bayes classifier

Data files:
params-5.txt- the 5-means for each class
params-5_151(3).txt- class mean file for first 151 classes and 5 means
testparams-5.txt- test image mean file for filtered images
correct.txt- file containing data on correctly classified images and accuracy