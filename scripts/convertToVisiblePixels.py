#!/usr/bin/python

import sys
import cv2
import os
import numpy as np

if len(sys.argv)<4:
    print('This scripts converts png images with bitwise pixel labeling into a human visible coloring. Also creates a key.')
    print('Usage: '+sys.argv[0]+' inDirectory outDirectory bit1name bit1color bit2name bit2color ... [-b (use 0x800000 boundary delimeter)]')
    exit()

def getColor(s):
    s=s.lower()
    if s=='blue' or s=='b':
        return (255,0,0)
    elif s=='green' or s=='g':
        return (0,255,0)
    elif s=='red' or s=='r':
        return (0,0,255)
    elif s=='cyan' or s=='c':
        return (255,255,0)
    elif s=='yellow' or s=='y':
        return (0,255,255)
    elif s=='magenta' or s=='purple' or s=='m' or s=='p':
        return (255,0,255)
    elif s=='white' or s=='w':
        return (255,255,255)
    elif s=='gray':
        return (155,155,155)

inDir = sys.argv[1]
outDir = sys.argv[2]
if not os.path.exists(outDir):
    os.makedirs(outDir)

boundaryD = 0x80 #in red channel
useBoundary=False

curBit = 0x01 #in blue channel
delimeters = []
out = open(outDir+'/key.txt','w')
for i in range(3,len(sys.argv),2):
    if (sys.argv[i] == '-b'):
        useBoundary=True
        break
    delimeters.append((curBit,sys.argv[i],getColor(sys.argv[i+1])))
    curBit = curBit<<1
    out.write(sys.argv[i]+':\t'+sys.argv[i+1]+'\n')
out.close()

for root, dirs, files in os.walk(inDir):
    for f in files:
        fs = f.split('.')
        if fs[1]=='png':
            print ('doing '+f)
            img = cv2.imread(root+'/'+f)
            show = np.empty_like(img)
            for r in range(img.shape[0]):
                for c in range(img.shape[1]):
                    colorCount=0
                    red=0
                    green=0
                    blue=0
                    for (bit, label, color) in delimeters:
                        if bit&img[r,c,0]:
                            colorCount+=1
                            blue+=color[0]
                            green+=color[1]
                            red+=color[2]
                        #assert(img[r,c,0] ^ (r==0 and b==0 and g==0))
                    if boundaryD&img[r,c,2]:
                        red+=255*colorCount
                        green+=255*colorCount
                        blue+=255*colorCount
                        colorCount *= 2
                    if colorCount>0:
                        red/=colorCount
                        green/=colorCount
                        blue/=colorCount
                    #print (img[r,c],(b,g,r))
                    show[r,c,0]=blue
                    show[r,c,1]=green
                    show[r,c,2]=red
                    #print (show[r,c],(b,g,r))

            cv2.imwrite(outDir+'/'+f,show)

