//
// Created by davidxu on 22-8-2.
//

#include "PPolynomial.cuh"

template<int Degree,int PolyCount>
class ConfirmedPPolynomial{
public:
    StartingPolynomial<Degree> polys[PolyCount];
    __host__ __device__ ConfirmedPPolynomial(void){}

    __host__ __device__ ConfirmedPPolynomial(const ConfirmedPPolynomial& cpy){
        for(int i=0;i<PolyCount;++i)
            polys[i]=cpy.polys[i];
    }

    __host__ __device__ ~ConfirmedPPolynomial(void){}

    __host__ __device__ ConfirmedPPolynomial scale(const float& s) const{
        ConfirmedPPolynomial ret;
        for(int i=0;i<PolyCount;++i)
            ret.polys[i]=polys[i].scale(s);
        return ret;
    }

    __host__ __device__ ConfirmedPPolynomial shift(const float& t) const{
        ConfirmedPPolynomial ret;
        for(int i=0;i<PolyCount;++i){
            ret.polys[i]=polys[i].shift(t);
        }
        return ret;
    }

    __host__ __device__ ConfirmedPPolynomial(const PPolynomial<Degree>& cpy){
        int tp=PolyCount<cpy.polyCount?PolyCount:cpy.polyCount;
        for(int i=0;i<tp;++i){
            polys[i]=cpy.polys[i];
        }
    }

};

template<int Degree,int PolyCount>
__device__ float value(ConfirmedPPolynomial<Degree,PolyCount> *cp,const float &val){
    float res=0;
    for(int i=0;i<PolyCount && val > cp->polys[i].start;++i){
        float temp=1;
        float v=0;
        for(int j=0;j<=Degree;++j){
            v+=temp * cp->polys[i].p.coefficients[j];
            temp*=val;
        }
        res+=v;
    }
    return res;
}