//g++ -g -std=c++11 weightPixelGT.cpp -lopencv_core -lopencv_imgcodecs -lopencv_imgproc -o weightPixelGT

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/imgcodecs.hpp>
#include <vector>
#include <map>
#include <tuple>
#include <limits>
#include <iostream>

#define INT_POS_INFINITY (numeric_limits<int>::max())

using namespace std;
using namespace cv;

/////////////////////////////////////////////////////////
//Meijster distance <http://fab.cba.mit.edu/classes/S62.12/docs/Meijster_distance.pdf>
//This can be parallelized. Should probably flip from column to row first
int f(int x, int i, int y, const Mat& g);
int SepPlusOne(int i, int u, int y, const Mat& g);
void computeDistanceMap(const Mat& img, const Mat& ccMap, int capDist Mat& out)
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
                if (d > ccMaxDists[ccMap.at<unsigned short>(y,u)])
                    ccMaxDists[ccMap.at<unsigned short>(y,u)] = d;
            }
            if (u==t[q])
                q--;
        }
    }
    //cout<<"max d: "<<maxDist<<endl;

    for (int x=0; x<=img.cols-1; x++)
        for (int y=0; y<=img.rows-1; y++)
            out.at<unsigned char>(y, x) = 255 * out.at<unsigned char>(y, x)/(ccMaxDists[ccMap.at<unsigned short>(y,x)]+0.0);
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


/*
   Get CC
   dialate the inverted image (make gt area larger)
   invert image
   Run the distance transform on inverted image (gets distance from background, aka the orders) (by CC?)
   Normalize weights by CC
   */

//this assumes a binary image. >0 on, 0 off
Mat weight(const Mat& img, int dilate_size, int capDist)
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

    vector<tuple<int,int,int,int> > ccBounds;//minX,minY,maxX,maxY
    Mat ccMap = getCCs(dilated,&ccBounds);
    

    Mat distanceMap = Mat::zeros(img.size(), img.type());
    computeDistanceMap(dilated, ccMap, capDist, distanceMap);
    return distanceMap;
}

int main(int argc, char** argv)
{
    int dilate=5;
    int capDist=10;
    string inFile = argv[1];
    string outFile = argv[2];
    Mat in = imread(inFile,0);//CV_LOAD_IMAGE_GRAYSCALE);
    assert(in.channels()==1);
    Mat out = weight(in,dilate,capDist);
    imwrite(outFile,out);
    return 0;
}
