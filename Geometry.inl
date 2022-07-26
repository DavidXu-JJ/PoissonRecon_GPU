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

template<class Real>
double SquareLength(const Point3D<Real>& p){return p.coords[0]*p.coords[0]+p.coords[1]*p.coords[1]+p.coords[2]*p.coords[2];}

template<class Real>
double Length(const Point3D<Real>& p){return sqrt(SquareLength(p));}

template<class Real>
double SquareDistance(const Point3D<Real>& p1,const Point3D<Real>& p2){
    return (p1.coords[0]-p2.coords[0])*(p1.coords[0]-p2.coords[0])+(p1.coords[1]-p2.coords[1])*(p1.coords[1]-p2.coords[1])+(p1.coords[2]-p2.coords[2])*(p1.coords[2]-p2.coords[2]);
}

template<class Real>
double Distance(const Point3D<Real>& p1,const Point3D<Real>& p2){return sqrt(SquareDistance(p1,p2));}

template <class Real>
void CrossProduct(const Point3D<Real>& p1,const Point3D<Real>& p2,Point3D<Real>& p){
    p.coords[0]= p1.coords[1]*p2.coords[2]-p1.coords[2]*p2.coords[1];
    p.coords[1]=-p1.coords[0]*p2.coords[2]+p1.coords[2]*p2.coords[0];
    p.coords[2]= p1.coords[0]*p2.coords[1]-p1.coords[1]*p2.coords[0];
}
template<class Real>
void EdgeCollapse(const Real& edgeRatio,std::vector<TriangleIndex>& triangles,std::vector< Point3D<Real> >& positions,std::vector< Point3D<Real> >* normals){
    int i,j,*remapTable,*pointCount,idx[3];
    Point3D<Real> p[3],q[2],c;
    double d[3],a;
    double Ratio=12.0/sqrt(3.0);	// (Sum of Squares Length / Area) for and equilateral triangle

    remapTable=new int[positions.size()];
    pointCount=new int[positions.size()];
    for(i=0;i<int(positions.size());i++){
        remapTable[i]=i;
        pointCount[i]=1;
    }
    for(i=int(triangles.size()-1);i>=0;i--){
        for(j=0;j<3;j++){
            idx[j]=triangles[i].idx[j];
            while(remapTable[idx[j]]<idx[j]){idx[j]=remapTable[idx[j]];}
        }
        if(idx[0]==idx[1] || idx[0]==idx[2] || idx[1]==idx[2]){
            triangles[i]=triangles[triangles.size()-1];
            triangles.pop_back();
            continue;
        }
        for(j=0;j<3;j++){
            p[j].coords[0]=positions[idx[j]].coords[0]/pointCount[idx[j]];
            p[j].coords[1]=positions[idx[j]].coords[1]/pointCount[idx[j]];
            p[j].coords[2]=positions[idx[j]].coords[2]/pointCount[idx[j]];
        }
        for(j=0;j<3;j++){
            q[0].coords[j]=p[1].coords[j]-p[0].coords[j];
            q[1].coords[j]=p[2].coords[j]-p[0].coords[j];
            d[j]=SquareDistance(p[j],p[(j+1)%3]);
        }
        CrossProduct(q[0],q[1],c);
        a=Length(c)/2;

        if((d[0]+d[1]+d[2])*edgeRatio > a*Ratio){
            // Find the smallest edge
            j=0;
            if(d[1]<d[j]){j=1;}
            if(d[2]<d[j]){j=2;}

            int idx1,idx2;
            if(idx[j]<idx[(j+1)%3]){
                idx1=idx[j];
                idx2=idx[(j+1)%3];
            }
            else{
                idx2=idx[j];
                idx1=idx[(j+1)%3];
            }
            positions[idx1].coords[0]+=positions[idx2].coords[0];
            positions[idx1].coords[1]+=positions[idx2].coords[1];
            positions[idx1].coords[2]+=positions[idx2].coords[2];
            if(normals){
                (*normals)[idx1].coords[0]+=(*normals)[idx2].coords[0];
                (*normals)[idx1].coords[1]+=(*normals)[idx2].coords[1];
                (*normals)[idx1].coords[2]+=(*normals)[idx2].coords[2];
            }
            pointCount[idx1]+=pointCount[idx2];
            remapTable[idx2]=idx1;
            triangles[i]=triangles[triangles.size()-1];
            triangles.pop_back();
        }
    }
    int pCount=0;
    for(i=0;i<int(positions.size());i++){
        for(j=0;j<3;j++){positions[i].coords[j]/=pointCount[i];}
        if(normals){
            Real l=Real(Length((*normals)[i]));
            for(j=0;j<3;j++){(*normals)[i].coords[j]/=l;}
        }
        if(remapTable[i]==i){ // If vertex i is being used
            positions[pCount]=positions[i];
            if(normals){(*normals)[pCount]=(*normals)[i];}
            pointCount[i]=pCount;
            pCount++;
        }
    }
    positions.resize(pCount);
    for(i=int(triangles.size()-1);i>=0;i--){
        for(j=0;j<3;j++){
            idx[j]=triangles[i].idx[j];
            while(remapTable[idx[j]]<idx[j]){idx[j]=remapTable[idx[j]];}
            triangles[i].idx[j]=pointCount[idx[j]];
        }
        if(idx[0]==idx[1] || idx[0]==idx[2] || idx[1]==idx[2]){
            triangles[i]=triangles[triangles.size()-1];
            triangles.pop_back();
        }
    }

    delete[] pointCount;
    delete[] remapTable;
}
template<class Real>
void TriangleCollapse(const Real& edgeRatio,std::vector<TriangleIndex>& triangles,std::vector< Point3D<Real> >& positions,std::vector< Point3D<Real> >* normals){
    int i,j,*remapTable,*pointCount,idx[3];
    Point3D<Real> p[3],q[2],c;
    double d[3],a;
    double Ratio=12.0/sqrt(3.0);	// (Sum of Squares Length / Area) for and equilateral triangle

    remapTable=new int[positions.size()];
    pointCount=new int[positions.size()];
    for(i=0;i<int(positions.size());i++){
        remapTable[i]=i;
        pointCount[i]=1;
    }
    for(i=int(triangles.size()-1);i>=0;i--){
        for(j=0;j<3;j++){
            idx[j]=triangles[i].idx[j];
            while(remapTable[idx[j]]<idx[j]){idx[j]=remapTable[idx[j]];}
        }
        if(idx[0]==idx[1] || idx[0]==idx[2] || idx[1]==idx[2]){
            triangles[i]=triangles[triangles.size()-1];
            triangles.pop_back();
            continue;
        }
        for(j=0;j<3;j++){
            p[j].coords[0]=positions[idx[j]].coords[0]/pointCount[idx[j]];
            p[j].coords[1]=positions[idx[j]].coords[1]/pointCount[idx[j]];
            p[j].coords[2]=positions[idx[j]].coords[2]/pointCount[idx[j]];
        }
        for(j=0;j<3;j++){
            q[0].coords[j]=p[1].coords[j]-p[0].coords[j];
            q[1].coords[j]=p[2].coords[j]-p[0].coords[j];
            d[j]=SquareDistance(p[j],p[(j+1)%3]);
        }
        CrossProduct(q[0],q[1],c);
        a=Length(c)/2;

        if((d[0]+d[1]+d[2])*edgeRatio > a*Ratio){
            // Find the smallest edge
            j=0;
            if(d[1]<d[j]){j=1;}
            if(d[2]<d[j]){j=2;}

            int idx1,idx2,idx3;
            if(idx[0]<idx[1]){
                if(idx[0]<idx[2]){
                    idx1=idx[0];
                    idx2=idx[2];
                    idx3=idx[1];
                }
                else{
                    idx1=idx[2];
                    idx2=idx[0];
                    idx3=idx[1];
                }
            }
            else{
                if(idx[1]<idx[2]){
                    idx1=idx[1];
                    idx2=idx[2];
                    idx3=idx[0];
                }
                else{
                    idx1=idx[2];
                    idx2=idx[1];
                    idx3=idx[0];
                }
            }
            positions[idx1].coords[0]+=positions[idx2].coords[0]+positions[idx3].coords[0];
            positions[idx1].coords[1]+=positions[idx2].coords[1]+positions[idx3].coords[1];
            positions[idx1].coords[2]+=positions[idx2].coords[2]+positions[idx3].coords[2];
            if(normals){
                (*normals)[idx1].coords[0]+=(*normals)[idx2].coords[0]+(*normals)[idx3].coords[0];
                (*normals)[idx1].coords[1]+=(*normals)[idx2].coords[1]+(*normals)[idx3].coords[1];
                (*normals)[idx1].coords[2]+=(*normals)[idx2].coords[2]+(*normals)[idx3].coords[2];
            }
            pointCount[idx1]+=pointCount[idx2]+pointCount[idx3];
            remapTable[idx2]=idx1;
            remapTable[idx3]=idx1;
            triangles[i]=triangles[triangles.size()-1];
            triangles.pop_back();
        }
    }
    int pCount=0;
    for(i=0;i<int(positions.size());i++){
        for(j=0;j<3;j++){positions[i].coords[j]/=pointCount[i];}
        if(normals){
            Real l=Real(Length((*normals)[i]));
            for(j=0;j<3;j++){(*normals)[i].coords[j]/=l;}
        }
        if(remapTable[i]==i){ // If vertex i is being used
            positions[pCount]=positions[i];
            if(normals){(*normals)[pCount]=(*normals)[i];}
            pointCount[i]=pCount;
            pCount++;
        }
    }
    positions.resize(pCount);
    for(i=int(triangles.size()-1);i>=0;i--){
        for(j=0;j<3;j++){
            idx[j]=triangles[i].idx[j];
            while(remapTable[idx[j]]<idx[j]){idx[j]=remapTable[idx[j]];}
            triangles[i].idx[j]=pointCount[idx[j]];
        }
        if(idx[0]==idx[1] || idx[0]==idx[2] || idx[1]==idx[2]){
            triangles[i]=triangles[triangles.size()-1];
            triangles.pop_back();
        }
    }
    delete[] pointCount;
    delete[] remapTable;
}

