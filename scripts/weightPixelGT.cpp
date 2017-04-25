//g++ -std=c++11 -fopenmp weightPixelGT.cpp -lcaffe -lglog -l:libopencv_core.so.3.1 -l:libopencv_imgcodecs.so.3.1 -l:libopencv_imgproc.so.3.1 -lprotobuf -lboost_system -I ../include/ -L ../build/lib/ -o weightPixelGT

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/imgcodecs.hpp>
#include <vector>
#include <tuple>

using namespace std;
using namespace cv;

//////////ZHANG SKELTONIZATION///////////////////////////////////////
///////////from https://github.com/bsdnoobz/zhang-suen-thinning//////
void thinningIteration(cv::Mat& img, int iter)
{
    CV_Assert(img.channels() == 1);
    CV_Assert(img.depth() != sizeof(uchar));
    CV_Assert(img.rows > 3 && img.cols > 3);

    cv::Mat marker = cv::Mat::zeros(img.size(), CV_8UC1);

    int nRows = img.rows;
    int nCols = img.cols;

    if (img.isContinuous()) {
        nCols *= nRows;
        nRows = 1;
    }

    int x, y;
    uchar *pAbove;
    uchar *pCurr;
    uchar *pBelow;
    uchar *nw, *no, *ne;    // north (pAbove)
    uchar *we, *me, *ea;
    uchar *sw, *so, *se;    // south (pBelow)

    uchar *pDst;

    // initialize row pointers
    pAbove = NULL;
    pCurr  = img.ptr<uchar>(0);
    pBelow = img.ptr<uchar>(1);

    for (y = 1; y < img.rows-1; ++y) {
        // shift the rows up by one
        pAbove = pCurr;
        pCurr  = pBelow;
        pBelow = img.ptr<uchar>(y+1);

        pDst = marker.ptr<uchar>(y);

        // initialize col pointers
        no = &(pAbove[0]);
        ne = &(pAbove[1]);
        me = &(pCurr[0]);
        ea = &(pCurr[1]);
        so = &(pBelow[0]);
        se = &(pBelow[1]);

        for (x = 1; x < img.cols-1; ++x) {
            // shift col pointers left by one (scan left to right)
            nw = no;
            no = ne;
            ne = &(pAbove[x+1]);
            we = me;
            me = ea;
            ea = &(pCurr[x+1]);
            sw = so;
            so = se;
            se = &(pBelow[x+1]);

            int A  = (*no == 0 && *ne >= 1) + (*ne == 0 && *ea >= 1) +
                     (*ea == 0 && *se >= 1) + (*se == 0 && *so >= 1) +
                     (*so == 0 && *sw >= 1) + (*sw == 0 && *we >= 1) +
                     (*we == 0 && *nw >= 1) + (*nw == 0 && *no >= 1);
            int B  = *no + *ne + *ea + *se + *so + *sw + *we + *nw;
            int m1 = iter == 0 ? (*no * *ea * *so) : (*no * *ea * *we);
            int m2 = iter == 0 ? (*ea * *so * *we) : (*no * *so * *we);

            if (A == 1 && (B >= 2 && B <= 6) && m1 == 0 && m2 == 0)
                pDst[x] = 1;
        }
    }

    img &= ~marker;
}

/**
 * Function for thinning the given binary image
 *
 * Parameters:
 *              src  The source image, binary with range = [0,255], 0 being off
 *              dst  The destination image
 */
void thinning(const cv::Mat& src, cv::Mat& dst)
{
    dst = src.clone();

    cv::Mat prev = cv::Mat::zeros(dst.size(), CV_8UC1);
    cv::Mat diff;

    do {
        thinningIteration(dst, 0);
        thinningIteration(dst, 1);
        cv::absdiff(dst, prev, diff);
        dst.copyTo(prev);
    }
    while (cv::countNonZero(diff) > 0);

    dst *= 255;

    cleanNoiseFromEdges(dst);
}
/////////////////////////////////////////////////////////
//Meijster distance <http://fab.cba.mit.edu/classes/S62.12/docs/Meijster_distance.pdf>
//This can be parallelized. Should probably flip from column to row first
int f(int x, int i, int y, const Mat& g);
int SepPlusOne(int i, int u, int y, const Mat& g);
void computeDistanceMapForCC(const Mat& ccMap, int ccId, int minX, int minY, int maxX, int maxY, 
                            Mat& out, Mat& g, int* s, int* t)
{
    //Mat out(src.size(), src.type());
    int maxDist=0;
    //Mat g(ccMap.size(), ccMap.type());
    for (int x=minX; x<=maxX; x++)
    {
        if (ccMap.at<unsigned short>(0,x)!=ccId)
        {
            g.at<int>(0, x)=0;
        }
        else
        {
            g.at<int>(0, x)=INT_POS_INFINITY;//src.cols*src.rows;
        }

        for (int y=minY+1; y<=maxY; y++)
        {
            if (ccMap.at<unsigned short>(y,x)!=ccId)
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

        for (int y=maxY-1; y>=minY; y--)
        {
            if (g.at<int>((y+1), x)<g.at<int>(y, x))
            {
                if (g.at<int>((y+1), x) != INT_POS_INFINITY)
                    g.at<int>(y, x)=1+g.at<int>((y+1), x);
                else
                    g.at<int>(y, x) = INT_POS_INFINITY;
            }
        }

        /*if(x==20)
        {
            for (int y=0; y<src.rows; y++)
            {
                printf("%d .. %d\n",qGray(src.pixel(x,y)),g.at<int>(y, x));
            }
        }*/
    }

    int q;
    //int s[ccMap.cols];
    //int t[ccMap.cols];
    int w;
    for (int y=minY; y<=maxY; y++)
    {
        q=minX;
        s[minX]=minX;
        t[minX]=minX;
        for (int u=minX+1; u<=maxX; u++)
        {
            while (q>=minX && f(t[q],s[q],y,g) > f(t[q],u,y,g))
            {
                q--;
            }

            if (q<minX)
            {
                q=minX;
                s[minX]=u;
            }
            else
            {
                w = SepPlusOne(s[q],u,y,g);
                if (w<=maxX)
                {
                    q++;
                    s[q]=u;
                    t[q]=w;
                }
            }
        }

        for (int u=maxX; u>=minX; u--)
        {
            out.at<unsigned char>(y, u)= min(f(u,s[q],y,g),100);
            if (out.at<unsigned char>(y, u) > maxDist)
                maxDist = out.at<unsigned char>(y, u);
            if (u==t[q])
                q--;
        }
    }

    //Normalization for CC
    for (int x=minX; x<=maxX; x++)
        for (int y=minY; y<=maxY; y++)
            out.at<unsigned char>(y, x) = 255 * out.at<unsigned char>(y, x)/(maxDist+0.0);
}

int SepPlusOne(int i, int u, int y, const Mat& g)
{
    if (g.at<unsigned char>(y,u) == INT_POS_INFINITY)
        return INT_POS_INFINITY;
    return 1 + ((u*u)-(i*i)+g.at<unsigned char>(y,u)*g.at<unsigned char>(y,u)-(g.at<unsigned char>(y,i)*g.at<unsigned char>(y,i))) / (2*(u-i));
}
int f(int x, int i, int y, const Mat& g)
{
    if (g.at<unsigned char>(y,i)==INT_POS_INFINITY || x==INT_POS_INFINITY)
        return INT_POS_INFINITY;
    return (x-i)*(x-i) + g.at<unsigned char>(y,i)*g.at<unsigned char>(y,i);
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
                if (c < get<0>(ccBounds.at(cc-1)))
                    get<0>(ccBounds.at(cc-1)) = c;
                if (r < get<1>(ccBounds.at(cc-1)))
                    get<1>(ccBounds.at(cc-1)) = r;
                if (c > get<2>(ccBounds.at(cc-1)))
                    get<2>(ccBounds.at(cc-1)) = c;
                if (r > get<3>(ccBounds.at(cc-1)))
                    get<3>(ccBounds.at(cc-1)) = r;
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
Mat weight(const Mat& img, int dilate_size)
{
    Mat element = getStructuringElement( MORPH_ELLIPSE,
                                         Size( 2*dilate_size + 1, 2*dilate_size+1 ),
                                         Point( dilate_size, dilate_size ) );
    Mat dilated;
    dilate( img, dilated, element );

    vector<tuple<int,int,int,int> > ccBounds;//minX,minY,maxX,maxY
    Mat ccMap = getCCS(dilated,&ccBounds);
    

    Mat distanceMap(img.size(), img.type());
    Mat workG(img.size(), CV_32S); //for memory effiency
    int workS[img.cols];               // "
    int workT[img.cols];               // "
    for (int cc=1; cc<=numCCs; cc++)
    {
        computeDistanceMapForCC(ccMap,cc,get<0>(ccBounds[cc-1]),get<1>(ccBounds[cc-1]),get<2>(ccBounds[cc-1]),get<3>(ccBounds[cc-1]), distanceMap,workG,workS,workT);
    }
    return distanceMap;
}

int main(int argc, char** argv)
{
    string inFile = argv[0];
    string outFile = argv[1];
    Mat in = imread(inFile,CV_LOAD_IMAGE_GRAYSCALE);
    Mat out = weight(in,5);
    imwrite(outFile,out);
    return 0;
}
