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

#ifndef POLYNOMIAL_INCLUDED
#define POLYNOMIAL_INCLUDED

#include <vector>

template<int Degree>
class Polynomial{
public:
    float coefficients[Degree+1];

    Polynomial(void);
    template<int Degree2>
    Polynomial(const Polynomial<Degree2>& P);

    /**     calculate the value of f(t)	    */
    __host__ __device__ float operator()(const float& t) const;

    /**     calculate the definite integral tMin/tMax [f(x)dx]
      *     tMin can be bigger than tMax	*/
    float integral(const float& tMin,const float& tMax) const;

    int operator == (const Polynomial& p) const;
    int operator != (const Polynomial& p) const;
    int isZero(void) const;
    void setZero(void);

    template<int Degree2>
    Polynomial& operator  = (const Polynomial<Degree2> &p);
    Polynomial& operator += (const Polynomial& p);
    Polynomial& operator -= (const Polynomial& p);
    Polynomial  operator -  (void) const;
    Polynomial  operator +  (const Polynomial& p) const;
    Polynomial  operator -  (const Polynomial& p) const;

    /**     Polynomial multiplication	*/
    template<int Degree2>
    Polynomial<Degree+Degree2>  operator *  (const Polynomial<Degree2>& p) const;

    Polynomial& operator += (const float& s);
    Polynomial& operator -= (const float& s);
    Polynomial& operator *= (const float& s);
    Polynomial& operator /= (const float& s);
    Polynomial  operator +  (const float& s) const;
    Polynomial  operator -  (const float& s) const;
    /**     scale the coefficients	*/
    Polynomial  operator *  (const float& s) const;
    Polynomial  operator /  (const float& s) const;

    /**     100.0000 x^0 +100.0000 x^1 +100.0000 x^2 (original doesn't change)
      *     scale(10.0) =>
      *     100.0000 x^0 +10.0000 x^1 +1.0000 x^2                    */
    __host__ __device__ Polynomial scale(const float& s) const;

    /**     f(x) -> f(x-t)
      *     1.0000 x^0 +1.0000 x^1 +1.0000 x^2 +1.0000 x^3  =>
      *     1
      *     -1   +   x
      *     1    -   2x  +   x^2
      *     -1    +   3x  -   3x^2    +   x^3
      *     => 0.0000 x^0 +2.0000 x^1 -2.0000 x^2 +1.0000 x^3       */
    __host__ __device__ Polynomial shift(const float& t) const;

    /**     f(x) -> f'(x)
      *     1.0000 x^0 +1.0000 x^1 +1.0000 x^2 +1.0000 x^3
      *     => derivative
      *     1.0000 x^0 +2.0000 x^1 +3.0000 x^2	                    */
    Polynomial<Degree-1> derivative(void) const;

    /**     calculate indefinite integral and let C = 0
      *     f(x) -> /f(x)dx
      *     1.0000 x^0 +2.0000 x^1 +3.0000 x^2 +4.0000 x^3
      *     => integral
      *     0.0000 x^0 +1.0000 x^1 +1.0000 x^2 +1.0000 x^3 +1.0000 x^4	*/
    Polynomial<Degree+1> integral(void) const;

    /** output  */
    void printnl(void) const;

    /**     add *this with scale * p (original is changed)
      *     1.0000 x^0 +1.0000 x^1 +1.0000 x^2 +1.0000 x^3
      *     a->addScaled(*a,10);
      *     11.0000 x^0 +11.0000 x^1 +11.0000 x^2 +11.0000 x^3	    */
    Polynomial& addScaled(const Polynomial& p,const float& scale);

    /**     $out will be erased, $out will be assigned with -$in
      *     Polynomial<Degree>::Negate will take only x^0 to x^Degree item of -$in
      *     $out must be Polynomial<Degree> , $in can be arbitrary	*/
    static void Negate(const Polynomial& in,Polynomial& out);

    /**     $q will be erased, $q will be assigned with $p1 - $p2
      *     Polynomial<Degree>::Subtract will take only x^0 to x^$Degree item of $p1 and $p2
      *     $q must be Polynomial<Degree>, $p1 and $p2 can be arbitrary	*/
    static void Subtract(const Polynomial& p1,const Polynomial& p2,Polynomial& q);

    /**     $q will be erased, $q will be assigned with $w * $p
      *     Polynomial<Degree>::Scale will take only x^0 to x^$Degree item of $p
      *     $q must be Polynomial<Degree>, $p can be arbitrary	    */
    static void Scale(const Polynomial& p,const float& w,Polynomial& q);

    /**     $q will be erased, $q will be assigned with $w1 * $p1 + $w2 * $p2
      *     q.coefficients[i]=p1.coefficients[i]*w1+p2.coefficients[i]*w2	*/
    static void AddScaled(const Polynomial& p1,const float& w1,const Polynomial& p2,const float& w2,Polynomial& q);

    /**     $q will be erased, $q will be assigned with $p1 + $w2 * $p2	*/
    static void AddScaled(const Polynomial& p1,const Polynomial& p2,const float& w2,Polynomial& q);

    /**     $q will be erased, $q will be assigned with $w1 * $p1 + $p2	*/
    static void AddScaled(const Polynomial& p1,const float& w1,const Polynomial& p2,Polynomial& q);

    /**     solve the equations
      *     coefficients[0] x^0 + coefficients[1] x^1 + coefficients[2] x^2 + coefficients[3] x^3 = c
      *     roots will be erased, the real solution x will be saved in roots
      *     imaginary solution x will be eliminated                  */
    void getSolutions(const float& c,std::vector<float>& roots,const float& EPS) const;
};

#include "Polynomial.inl"
#endif // POLYNOMIAL_INCLUDED
