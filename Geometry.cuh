/*
Copyright (c) 2006, Michael Kazhdan and Matthew Bolitho
All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

Redistributions of source code must retain the above copyright notice, this list of
conditions and the following disclaimer. Redistributions in binary form must reproduce
the above copyright notice, this list of conditions and the following disclaimer
in the documentation and/or other materials provided with the distribution.

Neither the name of the Johns Hopkins University nor the names of its contributors
may be used to endorse or promote products derived from this software without specific
prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY
EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO THE IMPLIED WARRANTIES
OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT
SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
TO, PROCUREMENT OF SUBSTITUTE  GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
DAMAGE.
*/

#ifndef GEOMETRY_INCLUDED
#define GEOMETRY_INCLUDED
#include <math.h>
#include <vector>

template<class Real>
struct Point3D{
    Real coords[3];
    inline       Real& operator[] ( int i )       { return coords[i]; }
    inline const Real& operator[] ( int i ) const { return coords[i]; }
    __host__ __device__ Point3D(void){
        coords[0]=0;
        coords[1]=0;
        coords[2]=0;
    }
    __host__ __device__ Point3D(const Point3D<Real>& cpy){
        coords[0]=cpy.coords[0];
        coords[1]=cpy.coords[1];
        coords[2]=cpy.coords[2];
    }
    __host__ __device__ Point3D<Real>& operator = (const Point3D<Real> &cpy){
        coords[0]=cpy.coords[0];
        coords[1]=cpy.coords[1];
        coords[2]=cpy.coords[2];
        return *this;
    }
};

template<class Real>
double Length(const Point3D<Real>& p);

template<class Real>
__host__ __device__ double SquareLength(const Point3D<Real>& p);

template<class Real>
double Distance(const Point3D<Real>& p1,const Point3D<Real>& p2);

template<class Real>
__host__ __device__ double SquareDistance(const Point3D<Real>& p1,const Point3D<Real>& p2);

template <class Real>
void CrossProduct(const Point3D<Real>& p1,const Point3D<Real>& p2,Point3D<Real>& p);

/**
 *      (p[0][0], p[0][1])
 *      (p[1][0], p[1][1])
 */
class Edge{
public:
    double p[2][2];
    double Length(void) const{
        double d[2];
        d[0]=p[0][0]-p[1][0];
        d[1]=p[0][1]-p[1][1];

        return sqrt(d[0]*d[0]+d[1]*d[1]);
    }
};
/**     (p[0][0], p[0][1], p[0][2]),
 *      (p[1][0], p[1][1], p[1][2]),
 *      (p[2][0], p[2][1], p[2][2])     */
class Triangle{
public:
    double p[3][3];
    double Area(void) const{
        double v1[3],v2[3],v[3];
        for(int d=0;d<3;d++){
            v1[d]=p[1][d]-p[0][d];
            v2[d]=p[2][d]-p[0][d];
        }
        v[0]= v1[1]*v2[2]-v1[2]*v2[1];
        v[1]=-v1[0]*v2[2]+v1[2]*v2[0];
        v[2]= v1[0]*v2[1]-v1[1]*v2[0];
        return sqrt(v[0]*v[0]+v[1]*v[1]+v[2]*v[2])/2;
    }
    double AspectRatio(void) const{
        double d=0;
        int i,j;
        for(i=0;i<3;i++){
            for(i=0;i<3;i++)
                for(j=0;j<3;j++){d+=(p[(i+1)%3][j]-p[i][j])*(p[(i+1)%3][j]-p[i][j]);}
        }
        return Area()/d;
    }

};
class CoredPointIndex{
public:
    int index;
    char inCore;

    int operator == (const CoredPointIndex& cpi) const {return (index==cpi.index) && (inCore==cpi.inCore);};
};
class EdgeIndex{
public:
    int idx[2];
};
class CoredEdgeIndex{
public:
    CoredPointIndex idx[2];
};
class TriangleIndex{
public:
    int idx[3];
};
template<class Real>
void EdgeCollapse(const Real& edgeRatio,std::vector<TriangleIndex>& triangles,std::vector< Point3D<Real> >& positions,std::vector<Point3D<Real> >* normals);
template<class Real>
void TriangleCollapse(const Real& edgeRatio,std::vector<TriangleIndex>& triangles,std::vector<Point3D<Real> >& positions,std::vector<Point3D<Real> >* normals);

class CoredMeshData{
public:
    std::vector<Point3D<float> > inCorePoints;
    const static int IN_CORE_FLAG[3];
    virtual void resetIterator(void)=0;

    virtual int addOutOfCorePoint(const Point3D<float>& p)=0;
    virtual int addTriangle(const TriangleIndex& t,const int& icFlag=(IN_CORE_FLAG[0] | IN_CORE_FLAG[1] | IN_CORE_FLAG[2]))=0;

    virtual int nextOutOfCorePoint(Point3D<float>& p)=0;
    virtual int nextTriangle(TriangleIndex& t,int& inCoreFlag)=0;

    virtual int outOfCorePointCount(void)=0;
    virtual int triangleCount(void)=0;
};

class CoredVectorMeshData : public CoredMeshData{
    std::vector<Point3D<float> > oocPoints;
    std::vector<TriangleIndex> triangles;
    /**     use as iterator                 */
    int oocPointIndex,triangleIndex;
public:
    CoredVectorMeshData(void);

    void resetIterator(void);

    int addOutOfCorePoint(const Point3D<float>& p);
    int addTriangle(const TriangleIndex& t,const int& inCoreFlag=(CoredMeshData::IN_CORE_FLAG[0] | CoredMeshData::IN_CORE_FLAG[1] | CoredMeshData::IN_CORE_FLAG[2]));

    int nextOutOfCorePoint(Point3D<float>& p);
    int nextTriangle(TriangleIndex& t,int& inCoreFlag);

    /**     Count all, include iterated     */
    int outOfCorePointCount(void);
    int triangleCount(void);
};
#include "Geometry.inl"

#endif // GEOMETRY_INCLUDED
