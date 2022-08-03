//
// Created by davidxu on 22-8-3.
//

#ifndef GPU_POISSONRECON_CONFIRMEDSPARSEMATRIX_CUH
#define GPU_POISSONRECON_CONFIRMEDSPARSEMATRIX_CUH

#include "SparseMatrix.cuh"
#include "ConfirmedVector.cuh"

template<int Rows,int Rowsize,class T>
class ConfirmedSparseSymmetricMatrix{
public:
    MatrixEntry<T> m_ppElements[Rows][Rowsize];

    template<int bSize,class T2>
    __host__ __device__ ConfirmedVector<bSize,T2> Multiply( const ConfirmedVector<bSize,T2>& V ) const{
        ConfirmedVector<Rows,T2> R;

        for (int i=0; i<Rows; i++){
            for(int ii=0;ii<Rowsize;ii++){
                int j=this->m_ppElements[i][ii].N;
                R(i)+=this->m_ppElements[i][ii].Value * V.m_pV[j];
                R(j)+=this->m_ppElements[i][ii].Value * V.m_pV[i];
            }
        }
        return R;
    }

    template<int bSize,class T2>
    __host__ __device__ void Multiply( const ConfirmedVector<bSize,T2>& In, ConfirmedVector<bSize,T2>& Out ) const{
        Out.SetZero();
        for (int i=0; i<Rows; i++){
            const MatrixEntry<T> *temp=this->m_ppElements[i];
            const T2& in1=In.m_pV[i];
            T2& out1=Out.m_pV[i];
            for(int ii=0;ii<Rowsize;ii++){
                const MatrixEntry<T>& temp2=temp[ii];
                int j=temp2.N;
                T2 v=temp2.Value;
                out1+=v * In.m_pV[j];
                Out.m_pV[j]+=v * in1;
            }
        }
    }

    template<int bSize,class T2>
    __host__ __device__ static int Solve(const ConfirmedSparseSymmetricMatrix<Rows,Rowsize,T>& M,const ConfirmedVector<bSize,T2> &b,
                                         const int& iters,
                                         ConfirmedVector<bSize,T2> &solution,
                                         const float eps=1e-8,const int& reset=1)
    {
        ConfirmedVector<bSize,T2> Md;
        T2 alpha,beta,rDotR,bDotB;
        if(reset){
            solution.SetZero();
        }
        ConfirmedVector<bSize,T2> d=b-M.Multiply(solution);     // error vector
        ConfirmedVector<bSize,T2> r=d;     // error vector
        rDotR=r.Dot(r);                 // L2 distance of error vector
        bDotB=b.Dot(b);                 // L2 distance of b
        if(b.Dot(b)<=eps){
            solution.SetZero();
            return 0;
        }
        int i;
        for(i=0;i<iters;i++){
            T2 temp;
            M.Multiply(d,Md);           // vec Md = matrix M * vec d
            temp=d.Dot(Md);
            if(fabs(temp)<=eps){break;}
            alpha=rDotR/temp;
            r.SubtractScaled(Md,alpha);
            temp=r.Dot(r);
            if(temp/bDotB<=eps){break;}
            beta=temp/rDotR;
            solution.AddScaled(d,alpha);
            if(beta<=eps){break;}
            rDotR=temp;
            ConfirmedVector<bSize,T2>::Add(d,beta,r,d);
        }
        return i;
    }
};


#endif //GPU_POISSONRECON_CONFIRMEDSPARSEMATRIX_CUH
