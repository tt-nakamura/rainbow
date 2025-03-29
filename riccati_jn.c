#include<math.h>

void riccati_jn(double *psi, int n, double x)
// riccati-bessel function psi_n(x) of first kind
// defined as x times spherical bessel function j_n(x)
// input: n,x
// output: psi: 1d-array (length n+1)
//   psi_n(x) for n=0,1,...,n
// assume psi is allocated by caller
{
    int i;
    psi[0] = sin(x);
    psi[1] = psi[0]/x - cos(x);
    for(i=2; i<=n; i++)// upward recurrence
        psi[i] = (2*i-1)*psi[i-1]/x - psi[i-2];
}