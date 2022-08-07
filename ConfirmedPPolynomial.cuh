//
// Created by davidxu on 22-8-2.
//

#ifndef GPU_POISSONRECON_CONFIRMEDPPOLYNOMIAL_CUH
#define GPU_POISSONRECON_CONFIRMEDPPOLYNOMIAL_CUH

#include "PPolynomial.cuh"

template<int Degree,int PolyCount>
class ConfirmedPPolynomial{
public:
    StartingPolynomial<Degree> polys[PolyCount];
    __host__ __device__ ConfirmedPPolynomial(void){
        for(int i=0;i<PolyCount;++i){
            polys[i].start=0;
            for(int j=0;j<=Degree;++j){
                polys[i].p.coefficients[j]=0;
            }
        }
    }

    __host__ __device__ ConfirmedPPolynomial(const ConfirmedPPolynomial& cpy){
        for(int i=0;i<PolyCount;++i)
            polys[i]=cpy.polys[i];
    }

    __host__ __device__ ~ConfirmedPPolynomial(void){}

//    __host__ __device__ ConfirmedPPolynomial scale(const float& s) const{
//        ConfirmedPPolynomial ret;
//        for(int i=0;i<PolyCount;++i)
//            ret.polys[i]=polys[i].scale(s);
//        return ret;
//    }

    __host__ __device__ ConfirmedPPolynomial shift(const float& t) const{
        ConfirmedPPolynomial ret;
        for(int i=0;i<PolyCount;++i){
            ret.polys[i].start=polys[i].start+t;
            for(int k=0;k<=Degree;k++){
                float temp=1;
                for(int j=k;j>=0;j--){
                    ret.polys[i].p.coefficients[j]+=polys[i].p.coefficients[k]*temp;
                    temp*=-t*j;
                    temp/=(k-j+1);
                }
            }
        }
        return ret;
    }

    __host__ __device__ ConfirmedPPolynomial(const PPolynomial<Degree>& cpy){
        int tp=PolyCount<cpy.polyCount?PolyCount:cpy.polyCount;
        for(int i=0;i<tp;++i){
            polys[i].start=cpy.polys[i].start;
            for(int j=0;j<=Degree;j++){
                polys[i].p.coefficients[j]=cpy.polys[i].p.coefficients[j];
            }
//            polys[i]=cpy.polys[i];
        }
    }

    __host__ __device__ ConfirmedPPolynomial& operator =(const PPolynomial<Degree>& cpy){
        int tp=PolyCount<cpy.polyCount?PolyCount:cpy.polyCount;
        for(int i=0;i<tp;++i){
            polys[i].start=cpy.polys[i].start;
            for(int j=0;j<=Degree;j++){
                polys[i].p.coefficients[j]=cpy.polys[i].p.coefficients[j];
            }
//            polys[i]=cpy.polys[i];
        }
        return *this;
    }

};

template<int Degree,int PolyCount>
__device__ float value(const ConfirmedPPolynomial<Degree,PolyCount> &cp,const float &val){
    float res=0;
    for(int i=0;i<PolyCount && val > cp.polys[i].start;++i){
        float temp=1;
        float v=0;
        for(int j=0;j<=Degree;++j){
            v+=temp * cp.polys[i].p.coefficients[j];
            temp*=val;
        }
        res+=v;
    }
    return res;
}

#endif //GPU_POISSONRECON_CONFIRMEDPPOLYNOMIAL_CUH
