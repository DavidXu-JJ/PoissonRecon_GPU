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

#ifndef FACTOR_INCLUDED
#define FACTOR_INCLUDED

#define PI 3.1415926535897932384
#define SQRT_3 1.7320508075688772935

/**     get the angle(radian = degree * PI / 180) of the point (x,y)    */
double ArcTan2(const double& y,const double& x);

/**     get the angle(radian = degree * PI / 180) of the point (in[0], in[1])   */
double Angle(const double in[2]);

/**     $out will be erased, the angle of point (in[0], in[1]) will be halved,
  *     the distance between (0,0) and new (out[0], out[1]) will be
  *     sqrt(distance between (0,0) and (in[0], in[1]))                 */
void Sqrt(const double in[2],double out[2]);

/**     $out will be erased, (out[0], out[1]) = (in1[0] + in2[0], in1[1] + in2[1])  */
void Add(const double in1[2],const double in2[2],double out[2]);

/**     $out will be erased, (out[0], out[1]) = (in1[0] - in2[0], in1[1] - in2[1])  */
void Subtract(const double in1[2],const double in2[2],double out[2]);

void Multiply(const double in1[2],const double in2[2],double out[2]);
void Divide(const double in1[2],const double in2[2],double out[2]);

/**     Equation ( a0 x^0 + a1 x^1 = 0 ) , get roots x ( roots[0][0] + roots[0][1] * i )
  *     return the number of roots                                                  */
int Factor(double a1,double a0,double roots[1][2],const double& EPS);
/**     Equation ( a0 x^0 + a1 x^1 + a2 x^2 = 0 ),
  *     get roots x1 = ( roots[0][0] + roots[0][1] * i )
  *               x2 = ( roots[1][0] + roots[1][1] * i )                           */
int Factor(double a2,double a1,double a0,double roots[2][2],const double& EPS);
int Factor(double a3,double a2,double a1,double a0,double roots[3][2],const double& EPS);
int Factor(double a4,double a3,double a2,double a1,double a0,double roots[4][2],const double& EPS);

/**     get the solution of $dim equations
  *     suppose  the solution is x1, x2 = [1, 2]
  *     3 x1 + 7 x2 = 17
  *     8 x1 + 2 x2 = 12
  *     [    3   7   ][  x1   ]   =   [   17  ]
  *     [    8   2   ][  x2   ]       [   12  ]
  *     eqns[ dim * dim ]    =   { 3, 7, 8, 2 }
  *     values [ dim ]       =   { 17, 12 }
  *     =>
  *     solutions [ dim ]    =   { 1, 2 }
  *     if there is certain sole solution, return 1
  *     if there is no or multiple solutions, return 0                              */
int Solve(const double* eqns,const double* values,double* solutions,const int& dim);

#endif // FACTOR_INCLUDED