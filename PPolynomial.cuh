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

#ifndef P_POLYNOMIAL_INCLUDED
#define P_POLYNOMIAL_INCLUDED
#include <vector>
#include "Polynomial.cuh"

template<int Degree>
class StartingPolynomial{
public:
    Polynomial<Degree> p;
    double start;

    /**	    return a StartingPolynomial
      *	    new start is the bigger start
      *	    polynomials are multiplied  */
    template<int Degree2>
    __host__ __device__ StartingPolynomial<Degree+Degree2>  operator * (const StartingPolynomial<Degree2>& p) const;

    /**     start = start * s
      *     polynomial is scaled by s   */
    __host__ __device__ StartingPolynomial scale(const double& s) const;

    /**     start = start + t
      *     polynomial f(x) -> f(x-t)   */
    __host__ __device__ StartingPolynomial shift(const double& t) const;

    /**     big start is bigger */
    __host__ __device__ int operator < (const StartingPolynomial& sp) const;

    /**     v1 > v2 , return 1
      *     v1 < v2 , return -1
      *     v1 = v2 , return 0  */
    __host__ __device__ static int Compare(const void* v1,const void* v2);
};

template<int Degree>
class PPolynomial{
public:
    size_t polyCount;
    StartingPolynomial<Degree>* polys;

    __host__ __device__ PPolynomial(void);
    __host__ __device__ PPolynomial(const PPolynomial<Degree>& p);
    __host__ __device__ ~PPolynomial(void);

    __host__ __device__ PPolynomial& operator = (const PPolynomial& p);

    /**     return size of polys    */
    __host__ __device__ int size(void) const;

    /**     polyCount = size, polys is allocated    */
    __host__ __device__ void set(const size_t& size);

    /**     Note: this method will sort the elements in sps
      *     polys with the same start will be added together    */
    __host__ __device__ void set(StartingPolynomial<Degree>* sps,const int& count);

    /**     realloc the memory to expand the polys pointer memory
      *     the old content will be remained    */
    __host__ __device__ void reset(const size_t& newSize);


    /**     assume that StartPolynomial is sorted by set() function
      *     calculate f0(t) + f1(t) + f2(t) + ... + fn(t)
      *     StartPolynomial n+1's start >= t    */
    __host__ __device__ double operator()(const double& t) const;


    /**     calculate the definite integral, integral start from the p[i].start, not min(tMin, tMax)
      *     let end = max(tMin, tMax)
      *     p[0].start / end [f0(x)dx] + p[1].start / end [f1(x)dx] + ... + p[n].start / end [fn(x)dx]
      *     p[n+1].start >= min(tMin, tMax)
      *     tMin can be bigger than tMax    */
    __host__ __device__ double integral(const double& tMin,const double& tMax) const;


    /**     integral(polys[0].start,polys[polyCount-1].start)   */
    __host__ __device__ double Integral(void) const;

    template<int Degree2>
    __host__ __device__ PPolynomial<Degree>& operator = (const PPolynomial<Degree2>& p);

    __host__ __device__ PPolynomial  operator + (const PPolynomial& p) const;
    __host__ __device__ PPolynomial  operator - (const PPolynomial& p) const;

    /**     remain the start
      *     multiply every polynomial by p  */
    template<int Degree2>
    __host__ __device__ PPolynomial<Degree+Degree2> operator * (const Polynomial<Degree2>& p) const;

    /**     for i in *this.polys
      *         for j in p.polys
      *              new.polys = i * j      */
    template<int Degree2>
    __host__ __device__ PPolynomial<Degree+Degree2> operator * (const PPolynomial<Degree2>& p) const;


    __host__ __device__ PPolynomial& operator += (const double& s);
    __host__ __device__ PPolynomial& operator -= (const double& s);
    __host__ __device__ PPolynomial& operator *= (const double& s);
    __host__ __device__ PPolynomial& operator /= (const double& s);
    __host__ __device__ PPolynomial  operator +  (const double& s) const;
    __host__ __device__ PPolynomial  operator -  (const double& s) const;
    __host__ __device__ PPolynomial  operator *  (const double& s) const;
    __host__ __device__ PPolynomial  operator /  (const double& s) const;

    /**     merge the *this and scale*poly
      *     poly with the same start will be added together */
    __host__ __device__ PPolynomial& addScaled(const PPolynomial& poly,const double& scale);

    /**     scale every poly in *this
      *     every start will be start * s   */
    __host__ __device__ PPolynomial scale(const double& s) const;

    /**     shift every poly in *this
      *     every start + t                 */
    __host__ __device__ PPolynomial shift(const double& t) const;

    /**     polys.start remain the same
      *     polys are derived               */
    __host__ __device__ PPolynomial<Degree-1> derivative(void) const;

    /**     polys.start remain the same
      *     definite integral function
      *     polys[i].start / x [fi(t)dt]
      *     Code:
      *     q.polys[i].p=polys[i].p.integral();
      *     q.polys[i].p-=q.polys[i].p(q.polys[i].start);  */
    __host__ __device__ PPolynomial<Degree+1> integral(void) const;

    /**     polys with $start < min are added together, get a new poly
      *     solve
      *     a0 x^0 + a1 x^1 + ... + an x^n = c
      *     save all solution accord with ( min < root < max )  */
    __host__ __device__ void getSolutions(const double& c,std::vector<double>& roots,const double& EPS,const double& min=-DBL_MAX,const double& max=DBL_MAX) const;

    __host__ __device__ void printnl(void) const;

    __host__ __device__ PPolynomial<Degree+1> MovingAverage(const double& radius);
    __host__ __device__ static PPolynomial ConstantFunction(const double& width=0.5);
    /**     use to generate approximation to Gaussian filter    */
    __host__ __device__ static PPolynomial GaussianApproximation(const double& width=0.5);
    void write(FILE* fp,const int& samples,const double& min,const double& max) const;
};
#include "PPolynomial.inl"
#endif // P_POLYNOMIAL_INCLUDED