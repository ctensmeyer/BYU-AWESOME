//g++ -g -std=c++11 weightPixelGT.cpp -lopencv_core -lopencv_imgcodecs -lopencv_imgproc -o weightPixelGT

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/imgcodecs.hpp>
#include <vector>
#include <set>
#include <map>
#include <tuple>
#include <limits>
#include <iostream>

#define INT_POS_INFINITY (numeric_limits<int>::max())
#define FLOAT_POS_INFINITY (numeric_limits<float>::max())

using namespace std;
using namespace cv;

/////////////////////////////////////////////////////////
//Meijster distance <http://fab.cba.mit.edu/classes/S62.12/docs/Meijster_distance.pdf>
//This can be parallelized. Should probably flip from column to row first
int f(int x, int i, int y, const Mat& g);
int SepPlusOne(int i, int u, int y, const Mat& g);
void computeDistanceMap(const Mat& orig, const Mat& img, const Mat& ccMap, int capDist, Mat& out)
{
    map<int,int> ccMaxDists;
    ccMaxDists[0]=1;
    Mat g(img.size(), CV_32S);
    int s[img.cols];         
    int t[img.cols];        
    int maxDist=0;
    //Mat g(ccMap.size(), ccMap.type());
    for (int x=0; x<=img.cols-1; x++)
    {
        if (img.at<unsigned char>(0,x)==0)
        {
            g.at<int>(0, x)=0;
        }
        else
        {
            g.at<int>(0, x)=INT_POS_INFINITY;//src.cols*src.rows;
        }

        for (int y=0+1; y<=img.rows-1; y++)
        {
            if (img.at<unsigned char>(y,x)==0)
            {
                g.at<int>(y, x)=0;
            }
            else
            {
                if (g.at<int>((y-1), x) != INT_POS_INFINITY)
                    g.at<int>(y, x)=1+g.at<int>((y-1), x);
                else
                    g.at<int>(y, x) = INT_POS_INFINITY;
            }
        }

        for (int y=img.rows-1-1; y>=0; y--)
        {
            if (g.at<int>((y+1), x)<g.at<int>(y, x))
            {
                if (g.at<int>((y+1), x) != INT_POS_INFINITY)
                    g.at<int>(y, x)=1+g.at<int>((y+1), x);
                else
                    g.at<int>(y, x) = INT_POS_INFINITY;
            }
        }
    }

    int q;
   
 
    int w;
    for (int y=0; y<=img.rows-1; y++)
    {
        q=0;
        s[0]=0;
        t[0]=0;
        for (int u=0+1; u<=img.cols-1; u++)
        {
            while (q>=0 && f(t[q],s[q],y,g) > f(t[q],u,y,g))
            {
                q--;
            }

            if (q<0)
            {
                q=0;
                s[0]=u;
            }
            else
            {
                w = SepPlusOne(s[q],u,y,g);
                if (w<=img.cols-1)
                {
                    q++;
                    s[q]=u;
                    t[q]=w;
                }
            }
        }

        for (int u=img.cols-1; u>=0; u--)
        {
            unsigned char d = std::min(std::sqrt(f(u,s[q],y,g)),0.0+capDist);
            if (ccMap.at<unsigned short>(y,u)!=0)
            {
                out.at<unsigned char>(y, u)= d;
                //store max distance by CC
                if (d > ccMaxDists[ccMap.at<unsigned short>(y,u)])
                    ccMaxDists[ccMap.at<unsigned short>(y,u)] = d;
            }
            if (u==t[q])
                q--;
        }
    }
    //cout<<"max d: "<<maxDist<<endl;

    //Normalize CCs
    for (int x=0; x<=img.cols-1; x++)
        for (int y=0; y<=img.rows-1; y++)
            if (orig.at<unsigned char>(y,x)>0) //clip to original "on" pixels
                out.at<unsigned char>(y, x) = 255 * out.at<unsigned char>(y, x)/(ccMaxDists[ccMap.at<unsigned short>(y,x)]+0.0);
            else
                out.at<unsigned char>(y, x)=0;
}

int SepPlusOne(int i, int u, int y, const Mat& g)
{
    if (g.at<int>(y,u) == INT_POS_INFINITY)
        return INT_POS_INFINITY;
    return 1 + ((u*u)-(i*i)+g.at<int>(y,u)*g.at<int>(y,u)-(g.at<int>(y,i)*g.at<int>(y,i))) / (2*(u-i));
}
int f(int x, int i, int y, const Mat& g)
{
    if (g.at<int>(y,i)==INT_POS_INFINITY || x==INT_POS_INFINITY)
        return INT_POS_INFINITY;
    return (x-i)*(x-i) + g.at<int>(y,i)*g.at<int>(y,i);
}

/*
Mat getCCs(const Mat& src, vector<tuple<int,int,int,int> >* ccBounds)
{
    Mat ccMap(src.size(), CV_16U);
    int count = connectedComponents (src, ccMap, 8, CV_16U);
    ccBounds->resize(count-1);
    for (int r=0; r<ccMap.rows; r++)
        for (int c=0; c<ccMap.cols; c++)
        {
            int cc = ccMap.at<unsigned short>(r,c);
            if (cc>0)
            {
                if (c < get<0>(ccBounds->at(cc-1)))
                    get<0>(ccBounds->at(cc-1)) = c;
                if (r < get<1>(ccBounds->at(cc-1)))
                    get<1>(ccBounds->at(cc-1)) = r;
                if (c > get<2>(ccBounds->at(cc-1)))
                    get<2>(ccBounds->at(cc-1)) = c;
                if (r > get<3>(ccBounds->at(cc-1)))
                    get<3>(ccBounds->at(cc-1)) = r;
            }
        }
    return ccMap;
}
*/


/*
   Get CC
   dialate the image (make gt area larger)
   Run the distance transform on ("inverted") image (gets distance from background, aka the orders)
   Normalize CCs individually
   */

//this assumes a binary image. >0 on, 0 off
Mat recallWeight(const Mat& img, int dilate_size, int capDist)
{
    Mat element = getStructuringElement( MORPH_ELLIPSE,
                                         Size( 2*dilate_size + 1, 2*dilate_size+1 ),
                                         Point( dilate_size, dilate_size ) );
    Mat elementBig = getStructuringElement( MORPH_ELLIPSE,
                                         Size( 4*dilate_size + 1, 4*dilate_size+1 ),
                                         Point( 2*dilate_size, 2*dilate_size ) );
    Mat dilated;
    if (dilate_size>0)
    {
        /*
        //Close
        dilate( img, dilated, elementBig );
        erode( dilated, dilated, elementBig );
        //Dilate
        dilate( dilated, dilated, element );
        */
        dilate( img, dilated, element );
    }
    else
        dilated=img;

    //vector<tuple<int,int,int,int> > ccBounds;//minX,minY,maxX,maxY
    //Mat ccMap = getCCs(dilated,&ccBounds);
    Mat ccMap(dilated.size(), CV_16U);
    int count = connectedComponents (dilated, ccMap, 8, CV_16U);
    

    Mat distanceMap = Mat::zeros(img.size(), img.type());
    computeDistanceMap(img,dilated, ccMap, capDist, distanceMap);
    return distanceMap;
}



void writeWeights(Mat& out, int startR, int startC, int norm, const Mat& values, const Mat& trace)
{
    int r=startR;
    int c=startC;
    while(values.at<float>(r,c)>0)
    {
        out.at<unsigned char>(r,c) = 128 + (127*values.at<float>(r,c))/norm;
        int t = trace.at<int>(r,c);
        r = t%trace.rows;
        c = t/trace.rows;
        assert(sqrt(pow(r-startR,2)+pow(c-startC,2)) <= norm+1);
        assert(values.at<float>(r,c)<values.at<float>(startR,startC));
    }
}
void writeWeights(Mat& out, int r, int c, int norm, const Mat& values, const vector< set<int> >& trace)
{
    if(values.at<float>(r,c)>0)
    {
        out.at<unsigned char>(r,c) = min(255,(int)(128 + (127*values.at<float>(r,c))/norm));//min((unsigned char)(128 + (127*values.at<float>(r,c))/norm),out.at<unsigned char>(r,c));
        for (int t : trace.at(r+c*values.rows))
        {
            int newR = t%values.rows;
            int newC = t/values.rows;
            writeWeights(out,newR,newC,norm,values,trace);
            //assert(out.at<unsigned char>(0,0)==0);
        }
    }
}

Mat precWeight(const Mat& img, int maxSize)
{
    Mat ccMap;
    int count = connectedComponents (img, ccMap, 8, CV_16U);
    Mat waveMap = img.clone();//Mat::zeros(img.size(),CV_16UC3); //waveNum,trace,cc
    waveMap = 1-(waveMap/255);
    waveMap.convertTo(waveMap,CV_32F);
    waveMap*=FLOAT_POS_INFINITY;
    //vector<Mat> channels(3);
    //channels.at(0)=waveMap;
    //channels.at(1)=Mat(img.size(),CV_32U);//trace x+y*cols
    //channels.at(2)=ccMap;
    //merge(channels,waveMap);
    //Mat trace = Mat(img.size(),CV_32S);//trace x+y*cols
    
    //map<int,set<int> > trace;
    vector< set<int> > trace(img.cols*img.rows);

    Mat ret(img.size(), CV_8U);
    ret=128;

    int minR=0;
    int maxR=img.rows;
    int minC=0;
    int maxC=img.cols;
    int nMinR=minR;
    int nMaxR=maxR;
    int nMinC=minC;
    int nMaxC=maxC;
    for (int wave=1; wave<=maxSize; wave++)
    {
        set<tuple<int,int,float> > colliding;
        for (int r=minR; r<maxR; r++)
            for (int c=minC; c<maxC; c++)
            {
                if (waveMap.at<float>(r,c)<wave && waveMap.at<float>(r,c)>=wave-1)
                {
                    //Any INF neighbors?
                    //Any different CC neighbors?
                    for (int rd=-1; rd<=1; rd++)
                        for (int cd=-1; cd<=1; cd++)
                        {
                            if ((rd==0 && cd==0) || r+rd<0 || r+rd>=waveMap.rows || c+cd<0 || c+cd>=waveMap.cols)
                                continue;
                            float distance = waveMap.at<float>(r,c) + (rd==0||cd==0?1:1.41421356237);
                            bool collide=ccMap.at<unsigned short>(r+rd,c+cd)!=0 && ccMap.at<unsigned short>(r+rd,c+cd)!=ccMap.at<unsigned short>(r,c);
                            if (waveMap.at<float>(r+rd,c+cd)>=distance)
                            {//move wave front here
                                waveMap.at<float>(r+rd,c+cd)=distance;
                                ccMap.at<unsigned short>(r+rd,c+cd)=ccMap.at<unsigned short>(r,c);
                                //trace[r+rd + (c+cd)*img.rows].insert(r + c*img.rows);
                                trace.at(r+rd + (c+cd)*img.rows).insert(r + c*img.rows);
                            }
                            if (collide)
                            {//wave front collision
                                float norm = max(waveMap.at<float>(r+rd,c+cd),waveMap.at<float>(r,c));
                                colliding.emplace(r,c,norm);
                                colliding.emplace(r+rd,c+cd,norm);
                                //writeWeights(ret,r,c,norm,waveMap,trace);
                                //writeWeights(ret,r+rd,c+cd,norm,waveMap,trace);
                            }
                        }
                    if (wave==1)
                    {
                        if (r>nMaxR)
                            nMaxR=r;
                        if (r<nMinR)
                            nMinR=r;
                        if (c>nMaxC)
                            nMaxC=c;
                        if (c<nMinC)
                            nMinC=c;
                    }
            //assert(ret.at<unsigned char>(0,0)==0);
                }
            }
        for (auto t : colliding)
        {
            writeWeights(ret,get<0>(t),get<1>(t),get<2>(t),waveMap,trace);
        }
        if (wave==1)
        {
            maxR=nMaxR+2;
            minR=nMinR-1;
            maxC=nMaxC+2;
            minC=nMinC-1;
        }
        //expand our RoI to allow expansion of wavefront
        maxR = min(img.rows,maxR+1);
        maxC = min(img.cols,maxC+1);
        minR = max(0,minR-1);
        minC = max(0,minC-1);

        cout<<"wavefront "<<wave<<" complete."<<endl;
    }

    return ret;
}

int main(int argc, char** argv)
{
    int dilate=3;
    int capDist=15;
    int maxDist=10;
    string inFile = argv[1];
    string outRecallFile = argv[2];
    string outPrecFile = argv[3];
    Mat in = imread(inFile,0);//CV_LOAD_IMAGE_GRAYSCALE);
    assert(in.channels()==1);
    Mat outPrec = precWeight(in,maxDist);
    imwrite(outPrecFile,outPrec);
    Mat outRecall = recallWeight(in,dilate,capDist);
    imwrite(outRecallFile,outRecall);
    return 0;
}
