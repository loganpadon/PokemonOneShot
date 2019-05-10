# import the necessary packages
import math
import re
import argparse
import numpy as np

def getProb(colors,meancolors,normalization):
  p=1
  for color in colors:
    dist=3*256
    for mean in meancolors:
      cdist=math.sqrt(math.pow(color[0]-mean[0],2)+math.pow(color[1]-mean[1],2)+math.pow(color[2]-mean[2],2))
      if cdist<dist:
        dist=cdist
    if dist>0:
      p*=1/dist/normalization
  return p

def getParams(input):
  split=re.split("\[|\]",input)
  colors=[]
  norm=1
  for i in range(len(split)):
    if i%2==1:
      parts=re.split(" |,|\t",split[i])
      while(len(parts)>3):
        parts.remove("")
      for j in range(3):
        parts[j]=float(parts[j])
      colors.append(parts)
    else:
      if i==0:
        parts=split[i].split("\t")
        norm=float(parts[1])
      
  return colors,norm

def getTestParams(input):
  split=re.split("\[|\]",input)
  colors=[]
  clabel=0
  for i in range(len(split)):
    if i%2==1:
      parts=re.split(" |,|\t",split[i])
      while(len(parts)>3):
        parts.remove("")
      for j in range(3):
        parts[j]=float(parts[j])
      colors.append(parts)
    else:
      if i==0:
        parts=split[i].split("-")
        clabel=int(parts[0])
      
  return colors,clabel
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
#ap.add_argument("-cm", "--classmeanfile", required = True, help = "Path #to the file contining mean colors for each class")
ap.add_argument("-tm", "--testmeanfile", required = True, help = "Path #to the file contining mean colors for each test image")
args = vars(ap.parse_args())
 
# load the image and convert it from BGR to RGB so that
# we can dispaly it with matplotlib
fileOut="params-"+str(args["clusters"])+".txt"
norms=dict()
means=dict()
fp=open(args["classmeanfile"],"r")
for i in range(151):
  n=i+1
  s=fp.readline()
  colors,norm=getParams(s)
  norms[n]=norm
  means[n]=colors
count=0
correct=0
fp.close()
fp=open(args["testmeanfile"],"r")
for i in range(937):
  s=fp.readline()
  colors,cval=getTestParams(s)
  p=0
  label=0
  for j in range(151):
    n=j+1
    prob=getProb(colors,means[n],norms[n])
    if(prob>p):
      p=prob
      label=n
  if label==cval:
    correct+=1
  count+=1
  
fp.close()
print(correct/count)
print(count)
print(correct)