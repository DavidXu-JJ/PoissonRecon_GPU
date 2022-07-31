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

#ifndef FUNCTION_DATA_INCLUDED
#define FUNCTION_DATA_INCLUDED


#include "PPolynomial.cuh"
#include "BinaryNode.cuh"

template<class Real>
class FunctionValueTable{
    int start,size;
    Real* values;
public:
    FunctionValueTable(void);
    ~FunctionValueTable(void);
    template<int Degree>
    /**     $res is resolution, the discrete value of ppoly(i/res) in [0, 1] are saved in $values   */
    int setValues(const PPolynomial<Degree>& ppoly,const int& res);
    /**     return ppoly(idx/res), if value is not defined then return 0                            */
    inline Real operator[] (const int& idx);
};
template<int Degree,class Real>
class FunctionData{
    /**     whether to scale the dDotTable, d2DotTable with dotTable                                */
    int useDotRatios;
    int normalize;
public:
    const static int     DOT_FLAG;
    const static int   D_DOT_FLAG;
    const static int  D2_DOT_FLAG;
    const static int   VALUE_FLAG;
    const static int D_VALUE_FLAG;

    /**     res is resolution               */
    int depth,res,res2;
    double *dotTable,*dDotTable,*d2DotTable;
    double *valueTables,*dValueTables;
    PPolynomial<Degree> baseFunction;
    /**     derivative of $baseFunction     */
    PPolynomial<Degree-1> dBaseFunction;
    PPolynomial<Degree+1>* baseFunctions;

    FunctionData(void);
    ~FunctionData(void);

    /**     if   (flags &   DOT_FLAG) is True,   $dotTable    will be set
      *          (flags & D_DOT_FLAG) is True,   $dDotTable   will be set
      *          (flags & D2_DOT_FLAG) is True,  $d2DotTable  will be set
      *     $dotTable contains inner product of the baseFunctions
      *     size of these array is all [res * res]                       */
    virtual void   setDotTables(const int& flags);
    virtual void clearDotTables(const int& flags);

    /**     if   (flags &   VALUE_FLAG) is True, $valueTables    will be set
      *          (flags & D_VALUE_FLAG) is True, $dValueTables   will be set
      *     $valueTables[i*res2 -- i*res2+res2-1] is smoothed baseFunctions[i]
      *     discrete value from [0, 1]
      *     so is dValueTables.
      *     size of these array is all [res * res2]                     */
    virtual void   setValueTables(const int& flags,const double& smooth=0);
    virtual void clearValueTables(void);

    /**     assume maxDepth is 2, assume REAL is one dimension
      *     the center and width of $index is
      *     [0] 0.500000, 1.000000
      *     [1] 0.250000, 0.500000
      *     [2] 0.750000, 0.500000
      *     [3] 0.125000, 0.250000
      *     [4] 0.375000, 0.250000
      *     [5] 0.625000, 0.250000
      *     [6] 0.875000, 0.250000
      *     $normalize scale the $F function and assign it to $baseFunction, so that it has:
      *     [0] Value 1 at 0
      *     [1] Integral equal to 1
      *     [2] Square integral equal to 1
      *     scale, shift and normalize the $baseFunction by center and width of $index,
      *     then assign it to baseFunctions[$index]
      *     baseFunctions[$index].start = (start * width) + center
      *     e.g :    start in [-1, 1]
      *              will be reflected into [center - width, center + width]    */
    void set(const int& maxDepth,const PPolynomial<Degree>& F,const int& normalize,const int& useDotRatios=1);

    /**     <F1, F2> inner product      */
    double   dotProduct(const double& center1,const double& width1,const double& center2,const double& width2) const;
    double  dDotProduct(const double& center1,const double& width1,const double& center2,const double& width2) const;
    double d2DotProduct(const double& center1,const double& width1,const double& center2,const double& width2) const;
};


#include "FunctionData.inl"
#endif // FUNCTION_DATA_INCLUDED