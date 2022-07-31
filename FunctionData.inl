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

////////////////////////
// FunctionValueTable //
////////////////////////
template<class Real>
FunctionValueTable<Real>::FunctionValueTable(void){
    start=-1;
    size=0;
    values=NULL;
}
template<class Real>
FunctionValueTable<Real>::~FunctionValueTable(void){
    if(values){delete[] values;}
    start=-1;
    size=0;
    values=NULL;
}
template<class Real>
template<int Degree>
int FunctionValueTable<Real>::setValues(const PPolynomial<Degree>& ppoly,const int& res){
    int j;
    if(values){delete[] values;}
    start=-1;
    size=0;
    values=NULL;
    for(j=0;j<res;j++){
        double x=double(j)/(res-1);
        if(x>ppoly.polys[0].start && x<ppoly.polys[ppoly.polyCount-1].start){
            if(start==-1){start=j;}
            size=j+1-start;
        }
    }
    if(size){
        values=new Real[size];
        for(j=0;j<size;j++){
            double x=double(j+start)/(res-1);
            values[j]=Real(ppoly(x));
        }
    }
    return size;
}
template<class Real>
inline Real FunctionValueTable<Real>::operator [] (const int& idx){
    int i=idx-start;
    if(i<0 || i>=size){return 0;}
    else{return values[i];}
}

//////////////////
// FunctionData //
//////////////////
template<int Degree,class Real>
const int FunctionData<Degree,Real>::DOT_FLAG=1;
template<int Degree,class Real>
const int FunctionData<Degree,Real>::D_DOT_FLAG=2;
template<int Degree,class Real>
const int FunctionData<Degree,Real>::D2_DOT_FLAG=4;
template<int Degree,class Real>
const int FunctionData<Degree,Real>::VALUE_FLAG=1;
template<int Degree,class Real>
const int FunctionData<Degree,Real>::D_VALUE_FLAG=2;

template<int Degree,class Real>
FunctionData<Degree,Real>::FunctionData(void){
    dotTable=dDotTable=d2DotTable=NULL;
    valueTables=dValueTables=NULL;
    res=0;
}

template<int Degree,class Real>
FunctionData<Degree,Real>::~FunctionData(void){
    if(res){
        delete[] dotTable;
        delete[] dDotTable;
        delete[] d2DotTable;
        delete[] valueTables;
        delete[] dValueTables;
    }
    dotTable=dDotTable=d2DotTable=NULL;
    valueTables=dValueTables=NULL;
    res=0;
}

template<int Degree,class Real>
void FunctionData<Degree,Real>::set(const int& maxDepth,
                                    const PPolynomial<Degree>& F,
                                    const int& normalize,
                                    const int& useDotRatios){
    this->normalize=normalize;
    this->useDotRatios=useDotRatios;

    depth=maxDepth;
    res=BinaryNode<double>::CumulativeCenterCount(depth);
    res2=(1<<(depth+1))+1;
    baseFunctions=new PPolynomial<Degree+1>[res];
    // Scale the function so that it has:
    // 0] Value 1 at 0
    // 1] Integral equal to 1
    // 2] Square integral equal to 1
    switch(normalize){
        case 2:
            baseFunction=F/sqrt((F*F).integral(F.polys[0].start,F.polys[F.polyCount-1].start));
            break;
        case 1:
            baseFunction=F/F.integral(F.polys[0].start,F.polys[F.polyCount-1].start);
            break;
        default:
            baseFunction=F/F(0);
    }
    dBaseFunction=baseFunction.derivative();
    double c1,w1;
    for(int i=0;i<res;i++){
        BinaryNode<double>::CenterAndWidth(i,c1,w1);
//        printf("original start:%lf\n",baseFunction.polys[1].start);
//        baseFunction.printnl();
//        printf("%lf,%lf\n",c1,w1);
        baseFunctions[i]=baseFunction.scale(w1).shift(c1);
//        printf("next start:%lf\n",baseFunctions[i].polys[1].start);
//        baseFunctions[i].printnl();
        // Scale the function so that it has L2-norm equal to one
        switch(normalize){
            case 2:
                baseFunctions[i]/=sqrt(w1);
                break;
            case 1:
                baseFunctions[i]/=w1;
                break;
        }
    }
}
template<int Degree,class Real>
void FunctionData<Degree,Real>::setDotTables(const int& flags){
    clearDotTables(flags);
    if(flags & DOT_FLAG){
        dotTable=new double[res*res];
        memset(dotTable,0,sizeof(double)*res*res);
    }
    if(flags & D_DOT_FLAG){
        dDotTable=new double[res*res];
        memset(dDotTable,0,sizeof(double)*res*res);
    }
    if(flags & D2_DOT_FLAG){
        d2DotTable=new double[res*res];
        memset(d2DotTable,0,sizeof(double)*res*res);
    }

    double t1,t2;
    t1=baseFunction.polys[0].start;
    t2=baseFunction.polys[baseFunction.polyCount-1].start;
    for(int i=0;i<res;i++){
        double c1,c2,w1,w2;
        BinaryNode<double>::CenterAndWidth(i,c1,w1);
        // map the function centers at 0 to its real position
        double start1	=t1*w1+c1;
        double end1		=t2*w1+c1;
        for(int j=0;j<=i;j++){
            BinaryNode<double>::CenterAndWidth(j,c2,w2);
            int idx1=i+res*j;
            int idx2=j+res*i;

            double start	=t1*w2+c2;
            double end		=t2*w2+c2;

            if(start<start1){start=start1;}
            if(end>end1)	{end=end1;}
            if(start>=end){continue;}

            BinaryNode<double>::CenterAndWidth(j,c2,w2);
            double dot=dotProduct(c1,w1,c2,w2);
            if(fabs(dot)<1e-15){continue;}
            if(flags & DOT_FLAG){dotTable[idx1]=dotTable[idx2]=dot;}
            if(useDotRatios){
                if(flags & D_DOT_FLAG){
                    dDotTable [idx1]= dDotProduct(c1,w1,c2,w2)/dot;
                    dDotTable [idx2]=-dDotTable[idx1];
                }
                if(flags & D2_DOT_FLAG){d2DotTable[idx1]=d2DotTable[idx2]=d2DotProduct(c1,w1,c2,w2)/dot;}
            }
            else{
                if(flags & D_DOT_FLAG){
                    dDotTable[idx1]= dDotProduct(c1,w1,c2,w2);
                    dDotTable[idx2]=-dDotTable[idx1];
                }
                if(flags & D2_DOT_FLAG){d2DotTable[idx1]=d2DotTable[idx2]=d2DotProduct(c1,w1,c2,w2);}
            }
        }
    }
}
template<int Degree,class Real>
void FunctionData<Degree,Real>::clearDotTables(const int& flags){
    if((flags & DOT_FLAG) && dotTable){
        delete[] dotTable;
        dotTable=NULL;
    }
    if((flags & D_DOT_FLAG) && dDotTable){
        delete[] dDotTable;
        dDotTable=NULL;
    }
    if((flags & D2_DOT_FLAG) && d2DotTable){
        delete[] d2DotTable;
        d2DotTable=NULL;
    }
}
template<int Degree,class Real>
void FunctionData<Degree,Real>::setValueTables(const int& flags,const double& smooth){
    clearValueTables();
    if(flags &   VALUE_FLAG){ valueTables=new double[res*res2];}
    if(flags & D_VALUE_FLAG){dValueTables=new double[res*res2];}
    PPolynomial<Degree+1> function;
    PPolynomial<Degree>  dFunction;
    for(int i=0;i<res;i++){
        if(smooth>0){
            function=baseFunctions[i].MovingAverage(smooth);
            dFunction=baseFunctions[i].derivative().MovingAverage(smooth);
        }
        else{
            function=baseFunctions[i];
            dFunction=baseFunctions[i].derivative();
        }
        for(int j=0;j<res2;j++){
            double x=double(j)/(res2-1);
            if(flags &   VALUE_FLAG){ valueTables[i*res2+j]= function(x);}
            if(flags & D_VALUE_FLAG){dValueTables[i*res2+j]=dFunction(x);}
        }
    }
}
template<int Degree,class Real>
void FunctionData<Degree,Real>::clearValueTables(void){
    if(valueTables){
        delete[] valueTables;
        valueTables=NULL;
    }
    if(dValueTables){
        delete[] dValueTables;
        dValueTables=NULL;
    }
}
template<int Degree,class Real>
double FunctionData<Degree,Real>::dotProduct(const double& center1,const double& width1,const double& center2,const double& width2) const{
    double r=fabs(baseFunction.polys[0].start);
    switch(normalize){
        case 2:
            return (baseFunction*baseFunction.scale(width2/width1).shift((center2-center1)/width1)).integral(-2*r,2*r)*width1/sqrt(width1*width2);
        case 1:
            return (baseFunction*baseFunction.scale(width2/width1).shift((center2-center1)/width1)).integral(-2*r,2*r)*width1/(width1*width2);
        default:
            return (baseFunction*baseFunction.scale(width2/width1).shift((center2-center1)/width1)).integral(-2*r,2*r)*width1;
    }
}
template<int Degree,class Real>
double FunctionData<Degree,Real>::dDotProduct(const double& center1,const double& width1,const double& center2,const double& width2) const{
    double r=fabs(baseFunction.polys[0].start);
    switch(normalize){
        case 2:
            return (dBaseFunction*baseFunction.scale(width2/width1).shift((center2-center1)/width1)).integral(-2*r,2*r)/sqrt(width1*width2);
        case 1:
            return (dBaseFunction*baseFunction.scale(width2/width1).shift((center2-center1)/width1)).integral(-2*r,2*r)/(width1*width2);
        default:
            return (dBaseFunction*baseFunction.scale(width2/width1).shift((center2-center1)/width1)).integral(-2*r,2*r);
    }
}
template<int Degree,class Real>
double FunctionData<Degree,Real>::d2DotProduct(const double& center1,const double& width1,const double& center2,const double& width2) const{
    double r=fabs(baseFunction.polys[0].start);
    switch(normalize){
        case 2:
            return (dBaseFunction*dBaseFunction.scale(width2/width1).shift((center2-center1)/width1)).integral(-2*r,2*r)/width2/sqrt(width1*width2);
        case 1:
            return (dBaseFunction*dBaseFunction.scale(width2/width1).shift((center2-center1)/width1)).integral(-2*r,2*r)/width2/(width1*width2);
        default:
            return (dBaseFunction*dBaseFunction.scale(width2/width1).shift((center2-center1)/width1)).integral(-2*r,2*r)/width2;
    }
}
