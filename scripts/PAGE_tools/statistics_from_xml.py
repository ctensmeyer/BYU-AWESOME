import sys
import parse_PAGE
import cv2
import line_extraction
import numpy as np
import os
import math
# import matplotlib.pyplot as plt

def meanAndStd(values):
    if len(values)==0:
         return 0,0
    mean_values=0.0
    for v in values:
         mean_values+=v
    mean_values/=len(values)
    std_values=0.0
    for v in values:
         std_values+=(v-mean_values)**2
    std_values/=len(values)
    return mean_values, math.sqrt(std_values)


if __name__ == "__main__":
    xml_directory = sys.argv[1]
    #img_directory = sys.argv[2]
    #output_directory = sys.argv[3]

    xml_filename_to_fullpath = {}
    for root, sub_folders, files in os.walk(xml_directory):
        for f in files:
            if not f.endswith(".xml"):
                continue
            f = f[:-len(".xml")]
            if f in xml_filename_to_fullpath:
                print "Error: this assumes no repeating files names: {} xml".format(f)

            xml_filename_to_fullpath[f] = root


    to_process = xml_filename_to_fullpath.keys()
    print "Number to be processed", len(to_process)

    #bl_count=0
    bl_length=[]
    bl_angle=[]
    #line_count=0
    line_numSegs=[]
    region_numLines=[]
    page_numSpacedSegs=[]
    #page_count=0
    page_blCount=[]
    page_blVertSpacing=[]

    for i, filename in enumerate(to_process):
        if i%10==0:
            print i
        xml_path = xml_filename_to_fullpath[filename]
        xml_path = os.path.join(xml_path, filename+".xml")
        try:
            #handle_single_image(xml_path, img_path, this_output_directory)
            xml_data = parse_PAGE.readXMLFile(xml_path)

            #page_count+=1
            averageSpacing=0
            spacingCount=0
            pageLineCount=0
            page_numSpacedSegs_sum=0
            lineBoundMin=-1
            lineBoundMax=-1
            linesInRegion=0
            for region in xml_data[0]['regions']:
                for i, line in enumerate(xml_data[0]['lines']):
                    if line['region_id'] != region['id']:
                        continue
                    #print(line['region_id'])
                    linesInRegion+=1
                    #line_mask = line_extraction.extract_baseline(img, line['baseline'])
                    pts = line['baseline']
                    new_pts = []
                    #minX=pts[0][0]
                    #maxX=pts[0][0]
                    minY=pts[0][1]
                    maxY=pts[0][1]
                    line_numSegs_sum=0
                    for i in range(len(pts)-1):
                        new_pts.append([pts[i], pts[i+1]])
                        #bl_count+=1
                        bl_length.append(math.sqrt((pts[i][0]-pts[i+1][0])**2 + (pts[i][1]-pts[i+1][1])**2))
                        bl_angle.append(math.atan2(pts[i+1][1]-pts[i][1], pts[i+1][0]-pts[i][0]))
                        if pts[i+1][1]<minY:
                             minY=pts[i+1][1]
                        if pts[i+1][1]>maxY:
                             maxY=pts[i+1][1]
                        #if pts[i+1][0]<minX:
                            #minX=pts[i+1][0]
                        #if pts[i+1][0]>maxX:
                            #maxX=pts[i+1][0]
                        line_numSegs_sum+=1
                    line_numSegs.append(line_numSegs_sum)
                    
                        
                    if (lineBoundMin<maxY and lineBoundMin>minY) or (lineBoundMax>minY and lineBoundMax<maxY) or \
                        (lineBoundMin<minY and lineBoundMax>maxY) or (lineBoundMin>minY and lineBoundMax<maxY):
                        #intesection, probably same line
                        page_numSpacedSegs_sum+=1
                        #lineBoundMin = min(lineBoundMin,minY)
                        #lineBoundMax = max(lineBoundMax,maxY)
                        lineBoundMin=minY
                        lineBoundMax=maxY
                        #print(pts[0],pts[1],lineBoundMin,lineBoundMax)
                        #print(filename)

                    else:
                        averageSpacing+=max(lineBoundMin-maxY, minY-lineBoundMax)
                        spacingCount+=1
                        lineBoundMin=minY
                        lineBoundMax=maxY
                        #line_count+=1
                        pageLineCount+=1

                region_numLines.append(linesInRegion)

            page_blVertSpacing.append(averageSpacing/spacingCount)
            page_blCount.append(pageLineCount)
            page_numSpacedSegs.append(page_numSpacedSegs_sum)
            

        except KeyboardInterrupt:
            raise
        except Exception as inst:
            out_str = xml_path+" Failed: "+str(type(inst))+" "+str(inst)
            print "".join(["*"]*len(out_str))
            print out_str
            print "".join(["*"]*len(out_str))

    mean_bl_length, std_bl_length = meanAndStd(bl_length)
    print('baseline length, mean: '+str(mean_bl_length)+',\tstd: '+str(std_bl_length))    
    
    mean_bl_angle, std_bl_angle = meanAndStd(bl_angle)
    print('baseline angle (deg), mean: '+str((180/math.pi) * mean_bl_angle)+',\tstd: '+str((180/math.pi) * std_bl_angle))    

    mean_line_numSegs, std_line_numSegs = meanAndStd(line_numSegs)
    print('segments per line (xml defined), mean: '+str(mean_line_numSegs)+',\tstd: '+str(std_line_numSegs))    

    mean_region_numLines, std_region_numLines = meanAndStd(region_numLines)
    print('lines per region (xml defined), mean: '+str(mean_region_numLines)+',\tstd: '+str(std_region_numLines))    
    
    mean_page_numSpacedSegs, std_page_numSpacedSegs = meanAndStd(page_numSpacedSegs)
    print('broken lines per page (est), mean: '+str(mean_page_numSpacedSegs)+',\tstd: '+str(std_page_numSpacedSegs))    
    
    mean_page_blCount, std_page_blCount = meanAndStd(page_blCount)
    print('baselines per page, mean: '+str(mean_page_blCount)+',\tstd: '+str(std_page_blCount))    
    
    mean_page_blVertSpacing, std_page_blVertSpacing = meanAndStd(page_blVertSpacing)
    print('avg vert baseline spacing per page, mean: '+str(mean_page_blVertSpacing)+',\tstd: '+str(std_page_blVertSpacing))    
    

