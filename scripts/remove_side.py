import numpy as np
import cv2
import math
import sys

def cropBlack(img,gt):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    median = np.median(gray)
    thresh = median*0.8

    cutTop=0
    while np.median(gray[cutTop,:]) < thresh:
        cutTop+=1
    
    cutBot=-1
    while np.median(gray[cutBot,:]) < thresh:
        cutBot-=1

    cutLeft=0
    while np.median(gray[:,cutLeft]) < thresh:
        cutLeft+=1
    
    cutRight=-1
    while np.median(gray[:,cutRight]) < thresh:
        cutRight-=1

    return img[cutTop:cutBot,cutLeft:cutRight], gt[cutTop:cutBot,cutLeft:cutRight], cutTop, -1*(cutBot+1), cutLeft, -1*(cutRight+1)

def getPageFold(img):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    kernel = np.array([2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,1.5,1,0.5,0,-25,-48,-25,0,0.5,1,1.5,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2]) / 800.0
    #kernel = np.array([4.0,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,3,2,1,0,-50,-96,-50,0,1,2,3,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4]) / 1600.0
    #kernel = np.array([2,2,2,2,2,2,2,1.5,1,0.5,0,-9,-16,-9,0,0.5,1,1.5,2,2,2,2,2,2,2])
    #kernel = np.array([5,5,5,5,5,5,4,3,2,1,0,-18,-34,-18,0,1,2,3,4,5,5,5,5,5,5])
    kernel = np.repeat(kernel,25,axis=0)
    edges = cv2.filter2D(gray,-1,kernel)
    #edges = np.absolute(edges)
    #off = (gray.shape[0]-edges.shape[0])/2
    #print off

    maxV = np.amax(edges)
    minV = np.amin(edges)
    print maxV, minV
    edges[:,:] = (edges[:,:]-minV)*(255.0/(maxV-minV))
    
    cv2.imwrite('testEdges.png',edges)
    """
    thresh=4000
    while True:
        rho_thetas = cv2.HoughLines(edges, 1,np.pi/180,thresh)
        if (len(rho_thetas[0])==1):
            break
        else if len(rho_thetas[0])==0:
            thresh*=1.5
        else:
            thresh/=2

    a = np.cos(rho_thetas[0][1])
    b = np.sin(rho_thetas[0][1])
    x0 = a*rho_thetas[0][0]
    y0 = b*rho_thetas[0][0]
    #x1 = int(x0 + 4000*(-b))
    #y1 = int(y0 + 4000*(a))
    #x2 = int(x0 - 4000*(-b))
    #y2 = int(y0 - 4000*(a))

    return x0
    """
    return edges

def removeCC(ccId, ccs, stats, removeFrom):
    #check=0
    for y in range(stats[ccId,cv2.CC_STAT_TOP],stats[ccId,cv2.CC_STAT_HEIGHT]+stats[ccId,cv2.CC_STAT_TOP]):
        for x in range(stats[ccId,cv2.CC_STAT_LEFT],stats[ccId,cv2.CC_STAT_WIDTH]+stats[ccId,cv2.CC_STAT_LEFT]):
            if ccs[y,x]==ccId:
                removeFrom[y,x]=0
    #            check+=1
    #assert check == stats[ccId,cv2.CC_STAT_AREA]
    #print 'removed ['+str(ccId)+'] of area '+str(check)


def getLen(line):
    return math.sqrt( (line['x1']-line['x2'])**2 + (line['y1']-line['y2'])**2 )

def convertToLineSegments(pred):
    ret=[]
    numLabels, labels, stats, cent = cv2.connectedComponentsWithStats(pred, 4, cv2.CV_32S)
    for l in range(1,numLabels):
        if stats[l,cv2.CC_STAT_WIDTH]>30:
            #xs=[]
            #ys=[]
            #for x in range(stats[l,cv2.CC_STAT_LEFT],stats[l,cv2.CC_STAT_WIDTH]+stats[l,cv2.CC_STAT_LEFT]):
            #    for y in range(stats[l,cv2.CC_STAT_TOP],stats[l,cv2.CC_STAT_HEIGHT]+stats[l,cv2.CC_STAT_TOP]):
            #        if labels[y,x]==l:
            #            xs.append(x)
            #            ys.append(y)
            #slope, intercept, r_value, p_value, std_err = stats.linregress(xs,ys)
            topLeft=-1
            topRight=-1
            for y in range(stats[l,cv2.CC_STAT_TOP],stats[l,cv2.CC_STAT_HEIGHT]+stats[l,cv2.CC_STAT_TOP]):
                if topLeft == -1 and labels[y,stats[l,cv2.CC_STAT_LEFT]]==l:
                    topLeft=y
                    if topRight != -1:
                        break
                if topRight == -1 and labels[y,stats[l,cv2.CC_STAT_LEFT]+stats[l,cv2.CC_STAT_WIDTH]-1]==l:
                    topRight=y
                    if topLeft != -1:
                        break

            botLeft=-1
            botRight=-1
            for y in range(stats[l,cv2.CC_STAT_HEIGHT]+stats[l,cv2.CC_STAT_TOP]-1,stats[l,cv2.CC_STAT_TOP],-1):
                if botLeft == -1 and labels[y,stats[l,cv2.CC_STAT_LEFT]]==l:
                    botLeft=y
                    if botRight != -1:
                        break
                if botRight == -1 and labels[y,stats[l,cv2.CC_STAT_LEFT]+stats[l,cv2.CC_STAT_WIDTH]-1]==l:
                    botRight=y
                    if botLeft != -1:
                        break
            ret.append({'x1':  stats[l,cv2.CC_STAT_LEFT],
                        'y1':  (topLeft+botLeft)/2,
                        'x2':  stats[l,cv2.CC_STAT_WIDTH]+stats[l,cv2.CC_STAT_LEFT]-1,
                        'y2':  (topRight+botRight)/2,
                        'cc':  l
                        })
        else:
            removeCC(l,labels,stats,pred)

    return ret, labels, stats

#file = sys.argv[1]

#predFile = '../../results/cbad_simple_base_weights_round_weighted_1_3/train/verbose/'+file+'/pred.png'
#origFile = '../../results/cbad_simple_base_weights_round_weighted_1_3/train/verbose/'+file+'/pred_on_original.png'

if len(sys.argv) != 4:
    print 'Usage: '+sys.argv[0]+' predImage origImage outPredImage'
    exit(0)

predFile = sys.argv[1]
origFile = sys.argv[2]
outName = sys.argv[3]


pred = cv2.imread(predFile,0)
orig = cv2.imread(origFile)

origCropped, predCropped, cropTop, cropBot, cropLeft, cropRight = cropBlack(orig,pred)

#clear pred on black areas
if cropTop>0:
    pred[:cropTop,:]=0
if cropBot>0:
    pred[-cropBot:,:]=0
if cropLeft>0:
    pred[:,:cropLeft]=0
if cropRight>0:
    pred[:,-cropRight:]=0


lines, ccs, ccStats = convertToLineSegments(predCropped)

meanLen=0
for line in lines:
    meanLen += getLen(line)
meanLen/=len(lines)
#print 'mean line: '+str(meanLen)

if cropLeft<4 or cropRight<4: #we can skip if we found black on both ends
    #pageLine = getPageFold(origCropped) too hard

    #vert hist of lines
    lineIm = np.zeros(predCropped.shape)
    for line in lines:
        if line is not None:
            cv2.line(lineIm, (line['x1'],line['y1']), (line['x2'],line['y2']), 1, 1)
    hist = np.sum(lineIm, axis=0)

    #construct linear filter based on mean line length
    kValues = [0.0]*int(meanLen*0.75)
    lenh=int(meanLen*0.75)/2
    for i in range(lenh):
        kValues[i] = -1.0*(lenh-i)
        kValues[-i] = (lenh-i)
    kernelLeftEdge = np.array(kValues)/lenh
    #kernelLeftEdge = np.array([-3,-3,-3,-3,-2,-2,-2.0,-2,-2,-1,0,1,2,2,2,2,2,3,3,3,3])/15.0
    #kernelLeftEdge = np.array([-6,-5,-5,-4,-4,-4,-3,-3,-3,-3,-2,-2,-2.0,-2,-2,-1,0,1,2,2,2,2,2,3,3,3,3,4,4,4,5,5,6])/21.0
    #kernelRightEdge = np.array([2.0,2,2,1,0,-1,-2,-2,-2])/5.0
    leftEdges = cv2.filter2D(hist,-1,kernelLeftEdge,None, (-1,-1), 0, cv2.BORDER_REPLICATE)
    #rightEdges = cv2.filter2D(hist,-1,kernelRightEdge)
    #val = filters.threshold_otsu(hist)

    maxV = np.amax(leftEdges)
    minV = np.amin(leftEdges)

    threshLeft = minV+(maxV-minV)*0.5
    threshRight = minV+(maxV-minV)*0.5

    leftPeaks = []
    hitLeft=False
    leftV=0
    rightPeaks = []
    hitRight=True
    rightV=-9999999
    for x in range(1,leftEdges.shape[0]-1):
        if leftEdges[x]>threshLeft and leftEdges[x]>leftEdges[x-1] and leftEdges[x]>leftEdges[x+1]:
            if hitRight:
                hitRight=False
                rightV=0
            if hitLeft:
                if leftEdges[x]>leftV:
                    leftV=leftEdges[x]
                    leftPeaks[-1]=x
            else:
                leftPeaks.append(x)
                hitLeft=True
                leftV=leftEdges[x]
        if leftEdges[x]<threshRight and leftEdges[x]<leftEdges[x-1] and leftEdges[x]<leftEdges[x+1]:
            if hitLeft:
                hitLeft=False
                leftV=0
            if hitRight:
                if leftEdges[x]<rightV:
                    rightV=leftEdges[x]
                    rightPeaks[-1]=x
            else:
                rightPeaks.append(x)
                hitRight=True
                rightV=leftEdges[x]

    #oldLeftPeaks=leftPeaks[:]
    #oldRightPeaks=rightPeaks[:]

    #prune peaks, assuming max left mataches min right and so on
    newLeftPeaks=[]
    newRightPeaks=[]
    while len(leftPeaks)>0 and len(rightPeaks)>0:
        maxLeft=leftPeaks[0]
        maxLeftV=leftEdges[maxLeft]
        for l in leftPeaks[1:]:
            if leftEdges[l] > maxLeftV:
                maxLeft=l
                maxLeftV=leftEdges[maxLeft]

        i=0
        while i < len(rightPeaks) and rightPeaks[i]<maxLeft:
            i+=1
        if i == len(rightPeaks):
            #then maxLeft has no matching peak
            newLeftPeaks.append(maxLeft)
            leftPeaks.remove(maxLeft)
            continue
        minRight=rightPeaks[i]
        minRightV=leftEdges[minRight]
        for r in rightPeaks[i:]:
            if leftEdges[r] < minRightV:
                minRight=r
                minRightV=leftEdges[minRight]

        if maxLeft>=minRight:
            print 'Error in peak pruning: '+predFile
            break

        newLeftPeaks.append(maxLeft)
        newRightPeaks.append(minRight)
        i=0
        while i < len(leftPeaks):
            if leftPeaks[i]>=maxLeft and leftPeaks[i]<=minRight:
                del leftPeaks[i]
            else:
                i+=1
        i=0
        while i < len(rightPeaks):
            if rightPeaks[i]>=maxLeft and rightPeaks[i]<=minRight:
                del rightPeaks[i]
            else:
                i+=1

    #pickup spare right peak
    if len(rightPeaks)>0:
        minRight=rightPeaks[0]
        minRightV=leftEdges[minRight]
        for r in rightPeaks[0:]:
            if leftEdges[r] < minRightV:
                minRight=r
                minRightV=leftEdges[minRight]
        newRightPeaks.append(minRight)

    if len(leftPeaks)>0:
        minLeft=leftPeaks[0]
        minLeftV=leftEdges[minLeft]
        for r in leftPeaks[0:]:
            if leftEdges[r] < minLeftV:
                minLeft=r
                minLeftV=leftEdges[minLeft]
        newLeftPeaks.append(minLeft)

    leftPeaks=sorted(newLeftPeaks)
    rightPeaks=sorted(newRightPeaks)


    #drawing
    """
    leftEdges = np.reshape(leftEdges,(1,leftEdges.shape[0]))
    leftEdges[:] = (leftEdges[:]-minV)*(255.0/(maxV-minV))
    origCropped[0:30,:,1]=leftEdges
    origCropped[0:30,:,0]=0
    origCropped[0:30,:,2]=0

    #for x in oldLeftPeaks:
    #    origCropped[0:30,x,2]=255
    #for x in oldRightPeaks:
    #    origCropped[0:30,x,0]=255
    for x in leftPeaks:
        origCropped[0:30,x,:]=0
        origCropped[0:30,x,2]=255
    for x in rightPeaks:
        origCropped[0:30,x,:]=0
        origCropped[0:30,x,2]=255
    """

    if len(leftPeaks)>2:
        print 'Warning: '+predFile+' post-proc may be in error. Too many sections starts, '+str(len(leftPeaks))+' detected.'
    if len(rightPeaks)>2:
        print 'Warning: '+predFile+' post-proc may be in error. Too many sections ends, '+str(len(leftPeaks))+' detected.'

    #check if up agains edge
    if cropLeft<4:  #Left side
        prune=-1
        keepLeft=leftPeaks[0]
        if len(rightPeaks)>1:
            if rightPeaks[0] < leftPeaks[0]:
                if rightPeaks[0] < rightPeaks[1]-leftPeaks[0]:
                    prune= rightPeaks[0]
                    keepLeft = leftPeaks[0]
            else:
                if leftPeaks[0]<meanLen*0.4 and rightPeaks[0]-leftPeaks[0] < rightPeaks[1]-leftPeaks[1]:
                    prune= rightPeaks[0]
                    keepLeft=leftPeaks[1]

        for i in range(len(lines)):
            line=lines[i]
            if (line['x1']<=meanLen/5 and getLen(line)<meanLen*0.75 and line['x2']<keepLeft) or (prune!=-1 and prune-line['x1']>line['x2']-prune):
                removeCC(line['cc'],ccs,ccStats,predCropped)
                lines[i]=None

    if cropRight<4: #Right side
        width = origCropped.shape[1]
        prune=-1
        keepRight = rightPeaks[-1]
        if len(leftPeaks)>1:
            print leftPeaks
            print rightPeaks
            if rightPeaks[-1] < leftPeaks[-1]:
                if width-leftPeaks[-1] < rightPeaks[-1]-leftPeaks[-2]:
                    prune= leftPeaks[-1]
                    keepRight = rightPeaks[-1]
            else:
                if rightPeaks[-1]-leftPeaks[-1] < rightPeaks[-2]-leftPeaks[-2]:
                    prune= leftPeaks[-1]
                    keepRight = rightPeaks[-2]

        for i in range(len(lines)):
            line=lines[i]
            
            if line is not None and ((line['x2']>=predCropped.shape[1]-(1+meanLen/5) and getLen(line)<meanLen*0.75 and line['x1']>keepRight) or (prune!=-1 and prune-line['x1']<line['x2']-prune)):
                removeCC(line['cc'],ccs,ccStats,predCropped)
                lines[i]=None

cv2.imwrite(outName,pred)

#draw

origCropped[:,0]=(255,0,0)
origCropped[:,-1]=(255,0,0)
origCropped[0,:]=(255,0,0)
origCropped[-1,:]=(255,0,0)


#origCropped[:,:,1]=pageLine

for line in lines:
    if line is not None:
        cv2.line(origCropped, (line['x1'],line['y1']), (line['x2'],line['y2']), (0,0,255), 7)
        #cv2.putText(origCropped, str(int(getLen(line))), (line['x1'],line['y1']), cv2.FONT_HERSHEY_PLAIN, 4, (255,0,0),3)

#cv2.imshow('res',orig)

#cv2.waitKey()
cv2.imwrite('test.png',orig)

