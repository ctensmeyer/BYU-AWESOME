
import sys
import cv2
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

ccSizeThreshold=200
lineLengthThreshold=200
intersectRange=1

img_path = sys.argv[1]

img = cv2.imread(img_path)
img = 255-img #np.subtract( np.array([[[255,255,255]]]), img) #invert, 255 is on
#plt.imshow(img)
#plt.show()

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ccs = np.zeros_like(gray)

def getLabel(l,labelMap):
    labelC=l
    while labelC in labelMap:
        if labelMap[labelC]==labelC
            return labelC
        labelC = labelMap[labelC]
    return labelC

#find connected components
labelMap = {}
ccPoints = {}
curLabel = 1
for y in range(gray.shape[0]):
    for x in range(gray.shape[1]):
        if gray[y,x]>0:
            if x>0 and gray[y,x-1]>0:
                ccs[y,x]=ccs[y,x-1]
                #if ccs[y,x] in ccCounts:
                ccPoints[ccs[y,x]].append((x,y))
                #else:
                #    ccPoints[ccs[y,x]]=[(x,y)]

                #combine CCs
                if y>0 and gray[y-1,x]>0:
                    labelC = getLabel(ccs[y-1,x],labelMap)
                    if labelC != ccs[y,x]:
                        labelMap[labelC]=ccs[y,x]
                        ccPoints[ccs[y,x]]+=ccPoints[labelC]
                        ccPoints[labelC]=None
                if y>0 and x>0 and gray[y-1,x-1]>0:
                    labelC = getLabel(ccs[y-1,x-1],labelMap)
                    if labelC != ccs[y,x]:
                        print(labelMap)
                        print(ccs[y,x])
                        print(labelC)
                        labelMap[labelC]=ccs[y,x]
                        ccPoints[ccs[y,x]]+=ccPoints[labelC]
                        ccPoints[labelC]=None
                if y>0 and x<gray.shape[1]-1 and gray[y-1,x+1]>0:
                    labelC = getLabel(ccs[y-1,x+1],labelMap)
                    if labelC != ccs[y,x]:
                        labelMap[labelC]=ccs[y,x]
                        ccPoints[ccs[y,x]]+=ccPoints[labelC]
                        ccPoints[labelC]=None

            elif y>0 and gray[y-1,x]>0:
                ccs[y,x]=getLabel(ccs[y-1,x],labelMap)
                ccPoints[ccs[y,x]].append((x,y))
            elif y>0 and x>0 and gray[y-1,x-1]>0:
                ccs[y,x]=getLabel(ccs[y-1,x-1],labelMap)
                ccPoints[ccs[y,x]].append((x,y))

                #combine CCs
                if y>0 and x<gray.shape[1]-1 and gray[y-1,x+1]>0:
                    labelMap[ccs[y-1,x+1]]=ccs[y,x]
                    ccPoints[ccs[y,x]]+=ccPoints[ccs[y-1,x+1]]
                    ccPoints[ccs[y-1,x+1]]=None

            elif y>0 and x<gray.shape[1]-1 and gray[y-1,x+1]>0:
                ccs[y,x]=ccs[y-1,x+1]
                ccPoints[ccs[y,x]].append((x,y))
            else:
                ccs[y,x]=curLabel
                curLabel+=1
                ccPoints[ccs[y,x]]=[(x,y)]
print('CC labeling done')

finalLines=[]

#regress lines on large enough CCs
for label in ccPoints:
    if label in labelMap:
        continue
    print('eval CC '+str(label))
    if len(ccPoints[label]) > ccSizeThreshold:
        while True:
            xs, ys = zip(*ccPoints[label])
            slope, intercept, r_value, p_value, std_err = stats.linregress(xs,ys)

            #check for CC intersections on wide lines
            intersections=[]
            for x in range(gray.shape[1]):
                cy = intercept + slope*x
                for y in range(int(cy-intersectRange), int(cy-intersectRange)):
                    if ccs[y,x]>0:
                        labelI = ccs[y,x]
                        while labelI in labelMap:
                            labelI = labelMap[labelI]
                        if labelI!=label:
                            print('intersecting CC '+str(labelI))
                            intersections.append(labelI)
                            labelMap[labelI]=label
                            ccPoints[label]+=ccPoints[labelI]

            #combine CC intersection
            if len(intersections)==0:
                break

        minX=min(xs)
        maxX=max(xs)
        if maxX-minX > lineLengthThresh:
            finalLines.append( ((minX,int(intercept + slope*minX)),(maxX,int(intercept + slope*maxX))) )

for line in finalLines:
    cv2.line(img,line[0],line[1],(0,255,0),2)

for line in finalLines:
    cv2.circle(img, line[0], 2, (255,0,0))
    cv2.circle(img, line[1], 2, (255,0,0))

plt.imshow(img)
plt.show()
