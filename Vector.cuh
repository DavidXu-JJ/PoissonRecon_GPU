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

#ifndef __VECTOR_HPP
#define __VECTOR_HPP

#define Assert assert
#include <assert.h>

template<class T>
class Vector
{
public:
    Vector();
    Vector( const Vector<T>& V );
    Vector( int N );
    Vector( int N, T* pV );
    ~Vector();

    __host__ __device__ const T& operator () (int i) const;
    __host__ __device__ T& operator () (int i);
    __host__ __device__ const T& operator [] (int i) const;
    __host__ __device__ T& operator [] (int i);

    /**     m_pV[0...m_N-1] are set to zero
     *      m_N doesn't change          */
    __host__ __device__ void SetZero();

    __host__ __device__ int Dimensions() const;

    /**     old memory will be cleared  */
    void Resize( int N );

    Vector operator * (const T& A) const;
    Vector operator / (const T& A) const;
    /**     the same size as *this      */
    Vector operator - (const Vector& V) const;
    Vector operator + (const Vector& V) const;

    __host__ __device__ Vector& operator *= (const T& A);
    __host__ __device__ Vector& operator /= (const T& A);
    __host__ __device__ Vector& operator += (const Vector& V);
    __host__ __device__ Vector& operator -= (const Vector& V);

    __host__ __device__ Vector& AddScaled(const Vector& V,const T& scale);
    __host__ __device__ Vector& SubtractScaled(const Vector& V,const T& scale);
    /**     $out will be the same size as V1    */
    __host__ __device__ static void Add(const Vector& V1,const T& scale1,const Vector& V2,const T& scale2,Vector& Out);
    __host__ __device__ static void Add(const Vector& V1,const T& scale1,const Vector& V2,Vector& Out);

    Vector operator - () const;

    Vector& operator = (const Vector& V);

    __host__ __device__ T Dot( const Vector& V ) const;

    __host__ __device__ T Length() const;

    __host__ __device__ T Norm( int Ln ) const;
    __host__ __device__ void Normalize();

    T* m_pV;
protected:
    int m_N;

};

template<class T>
__host__ void copySingleVector(Vector<T> *v_h,Vector<T> *&v_d){
    T *d_addr=NULL;
    int nByte=sizeof(T) * v_h->Dimensions();
    cudaMalloc((T**)&d_addr,nByte);
    cudaMemcpy(d_addr,v_h->m_pV,nByte,cudaMemcpyHostToDevice);
    T *h_addr = v_h->m_pV;
    v_h->m_pV = d_addr;
    cudaMalloc((Vector<T> **)&v_d,sizeof(Vector<T>));
    cudaMemcpy(v_d,v_h,sizeof(Vector<T>),cudaMemcpyHostToDevice);
    v_h->m_pV = h_addr;
}

template<class T>
__host__ void copyWholeVectorArray(Vector<T> *v_h,Vector<T> *&v_d,int size){
    int nByte;
    std::vector<T*> ptr_v;
    for(int i=0;i<size;++i){
        nByte=sizeof(T) * (v_h+i)->Dimensions();
        T *d_addr=NULL;
        cudaMalloc((T**)&d_addr,nByte);
        cudaMemcpy(d_addr,(v_h+i)->m_pV,nByte,cudaMemcpyHostToDevice);
        ptr_v.push_back((v_h+i)->m_pV);
        (v_h+i)->m_pV=d_addr;
    }

    nByte=sizeof(Vector<T>) * size;
    cudaMalloc((Vector<T>**)&v_d,nByte);
    cudaMemcpy(v_d,v_h,nByte,cudaMemcpyHostToDevice);

    for(int i=0;i<size;++i){
        (v_h+i)->m_pV=ptr_v[i];
    }
}


template<class T,int Dim>
class NVector
{
public:
    NVector();
    NVector( const NVector& V );
    NVector( int N );
    NVector( int N, T* pV );
    ~NVector();

    const T* operator () (int i) const;
    T* operator () (int i);
    const T* operator [] (int i) const;
    T* operator [] (int i);

    void SetZero();

    int Dimensions() const;

    /**     Alloc N*Dim*sizeof(T)   memory  */
    void Resize( int N );

    NVector operator * (const T& A) const;
    NVector operator / (const T& A) const;
    NVector operator - (const NVector& V) const;
    NVector operator + (const NVector& V) const;

    NVector& operator *= (const T& A);
    NVector& operator /= (const T& A);
    NVector& operator += (const NVector& V);
    NVector& operator -= (const NVector& V);

    NVector& AddScaled(const NVector& V,const T& scale);
    NVector& SubtractScaled(const NVector& V,const T& scale);
    static void Add(const NVector& V1,const T& scale1,const NVector& V2,const T& scale2,NVector& Out);
    static void Add(const NVector& V1,const T& scale1,const NVector& V2,				NVector& Out);

    NVector operator - () const;

    NVector& operator = (const NVector& V);

    T Dot( const NVector& V ) const;

    T Length() const;

    T Norm( int Ln ) const;
    void Normalize();

    T* m_pV;
protected:
    int m_N;

};


#include "Vector.inl"

#endif
