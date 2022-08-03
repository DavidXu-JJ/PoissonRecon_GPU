//
// Created by davidxu on 22-8-3.
//

#ifndef GPU_POISSONRECON_CONFIRMEDVECTOR_CUH
#define GPU_POISSONRECON_CONFIRMEDVECTOR_CUH


template<int Size,class T>
class ConfirmedVector
{
public:
    __host__ __device__ ConfirmedVector(void){}

    __host__ __device__ ConfirmedVector( const ConfirmedVector<Size,T>& V ){
        for(int i=0;i<Size;++i){
            m_pV[i]=V.m_pV[i];
        }
    }

    __host__ __device__ ConfirmedVector( const T *array){
        for(int i=0;i<Size;++i){
            m_pV[i]=array[i];
        }
    }

    __host__ __device__ const T& operator () (int i) const{
        return m_pV[i];
    }
    __host__ __device__ T& operator () (int i){
        return m_pV[i];
    }
    __host__ __device__ const T& operator [] (int i) const{
        return m_pV[i];
    }
    __host__ __device__ T& operator [] (int i){
        return m_pV[i];
    }

    /**     m_pV[0...m_N-1] are set to zero
     *      m_N doesn't change          */
    void SetZero(){
        for(int i=0;i<Size;++i){
            m_pV[i]=0;
        }
    }

    __host__ __device__ int Dimensions() const{
        return Size;
    }

    __host__ __device__ ConfirmedVector operator * (const T& A) const{
        ConfirmedVector ret;
        for (int i=0; i<Size; i++)
            ret.m_pV[i] = this->m_pV[i] * A;
        return ret;
    }
    __host__ __device__ ConfirmedVector operator / (const T& A) const{
        ConfirmedVector ret;
        for (int i=0; i<Size; i++)
            ret.m_pV[i] = this->m_pV[i] / A;
        return ret;
    }
    /**     the same size as *this      */
    __host__ __device__ ConfirmedVector operator - (const ConfirmedVector& V) const{
        ConfirmedVector ret;
        for (int i=0; i<Size; i++)
            ret.m_pV[i] = m_pV[i] - V.m_pV[i];
        return ret;
    }
    __host__ __device__ ConfirmedVector operator + (const ConfirmedVector& V) const{
        ConfirmedVector ret;
        for (int i=0; i<Size; i++)
            V.m_pV[i] = m_pV[i] + V.m_pV[i];
        return ret;
    }

    __host__ __device__ ConfirmedVector& operator *= (const T& A){
        for (int i=0; i<Size; i++)
            m_pV[i] *= A;
        return *this;
    }
    __host__ __device__ ConfirmedVector& operator /= (const T& A){
        for (int i=0; i<Size; i++)
            m_pV[i] /= A;
        return *this;
    }
    __host__ __device__ ConfirmedVector& operator += (const ConfirmedVector& V){
        for (int i=0; i<Size; i++)
            m_pV[i] += V.m_pV[i];
        return *this;
    }
    __host__ __device__ ConfirmedVector& operator -= (const ConfirmedVector& V){
        for (int i=0; i<Size; i++)
            m_pV[i] -= V.m_pV[i];
        return *this;
    }

    __host__ __device__ ConfirmedVector& AddScaled(const ConfirmedVector& V,const T& scale){
        for (int i=0; i<Size; i++)
            m_pV[i] += V.m_pV[i]*scale;
        return *this;
    }
    __host__ __device__ ConfirmedVector& SubtractScaled(const ConfirmedVector& V,const T& scale){
        for (int i=0; i<Size; i++)
            m_pV[i] -= V.m_pV[i]*scale;
        return *this;
    }
    /**     $out will be the same size as V1    */
    __host__ __device__ static void Add(const ConfirmedVector& V1,const T& scale1,const ConfirmedVector& V2,const T& scale2,ConfirmedVector& Out){
        for (int i=0; i<Size; i++)
            Out.m_pV[i]=V1.m_pV[i]*scale1+V2.m_pV[i]*scale2;
    }
    __host__ __device__ static void Add(const ConfirmedVector& V1,const T& scale1,const ConfirmedVector& V2,ConfirmedVector& Out){
        for (int i=0; i<Size; i++)
            Out.m_pV[i]=V1.m_pV[i]*scale1+V2.m_pV[i];
    }

    __host__ __device__ ConfirmedVector operator - () const{
        ConfirmedVector ret;
        for (int i=0; i<Size; i++)
            ret.m_pV[i] = -m_pV[i];
        return ret;
    }

    __host__ __device__ ConfirmedVector& operator = (const ConfirmedVector& V){
        for(int i=0;i<Size;++i){
            m_pV[i]=V.m_pV[i];
        }
        return *this;
    }

    __host__ __device__ T Dot( const ConfirmedVector& V ) const{
        T V0 = T(0);
        for (int i=0; i<Size; i++)
            V0 += m_pV[i]*V.m_pV[i];
        return V0;
    }

    __host__ __device__ T Length() const{
        T N = T(0);
        for (int i = 0; i<Size; i++)
            N += m_pV[i]*m_pV[i];
        return sqrt(N);
    }

    __host__ __device__ T Norm( size_t Ln ) const{
        T N = T(0);
        for (int i = 0; i<Size; i++)
            N += pow(m_pV[i], (T)Ln);
        return pow(N, (T)1.0/Ln);
    }
    __host__ __device__ void Normalize(){
        T N = 1.0f/Norm(2);
        for (int i = 0; i<Size; i++)
            m_pV[i] *= N;
    }

    T m_pV[Size];

};


#endif //GPU_POISSONRECON_CONFIRMEDVECTOR_CUH
