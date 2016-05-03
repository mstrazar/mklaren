#include "mex.h"
#include <math.h>
#include "matrix.h"

void qr2(double *M, double *Q, double *R)
{
/* perform QR decomposition on a matrix with two elements */
double x;
x = sqrt( M[0+2*0] * M[0+2*0] + M[1+2*0] * M[1+2*0] );
R[0+2*0] = x;
Q[0+2*0] = M[0+2*0] / x;
Q[1+2*0] = M[1+2*0] / x;
R[0+2*1] = ( M[0+2*0] * M[0+2*1] + M[1+2*0] * M[1+2*1] ) / x;
R[1+2*0] = 0.0;
Q[0+2*1] = M[0+2*1] - R[0+2*1] * Q[0+2*0];
Q[1+2*1] = M[1+2*1] - R[0+2*1] * Q[1+2*0];
R[1+2*1] = sqrt(Q[0+2*1] * Q[0+2*1] + Q[1+2*1] * Q[1+2*1] );
Q[0+2*1] /= R[1+2*1];
Q[1+2*1] /= R[1+2*1];
return;
}




void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
 
/*

% INPUT
% K  : kernel matrix n x n
% Y  : target vector n x d
% m  : maximal rank
% kappa : trade-off between approximation of K and prediction of Y (suggested: .99)
% centering : 1 if centering, 0 otherwise (suggested: 1)
% delta : number of columns of cholesky performed in advance (suggested: 40)
% tol : minimum gain at iteration (suggested: 1e-4)
%
% OUTPUT
% G : Cholesky decomposition -> K(P,P) is approximated by G*G'
% P : permutation matrix
% Q,R : QR decomposition of G (or center(G) if centering)
% error1 : tr(K-G*G')/tr(K) at each step of the decomposition
% error2 : ||Y-Q*Q'*Y||_F^2 / ||Y||_F^2 at each step of the decomposition
% predicted_gain : predicted gain before adding each column
% true_gain : actual gain after adding each column
%
% Copyright (c) Francis R. Bach, 2005.


*/
/* variable declarations */
double *K,*Y,tol,*tempref,*error1,*error2,*Q,*R,*D,*dJK,*dJY,*y;
double *Gcol,*DK,*GTG,*QTY,*Q2,*R2,*tempvec,*tempvec2,temp2,temp1,*meanY;
int n,d,m,centering,delta,maxcol,*P,nd,k,i,j,q,display;
double kappa,*G,*dJ,traceK,sumY,sumY2,mu,lambda,diagmax,temp,*predicted_gain,*true_gain;
int endloop,jast,tempint,kast,s,p,force_exit,kadv,newpivot,b;
double *M2,*Dadv,*A1,*A2,*A3,*tempA2,a,*Gbef,*GTGbef,*tempQ,*Rbef,*tempG,*tempGG,*tempA3;
double *Gbeftotal,*Rbeftotal,*QTYYTQ,*QTYYTQbef,*tempA4;
  
/* input */
K    = mxGetPr(prhs[0]);       /* kernel matrix : n x n */
n    = mxGetM(prhs[0]);
Y    = mxGetPr(prhs[1]);          /* target matrix : n x d */
d    = mxGetN(prhs[1]);
tempref = mxGetPr(prhs[2]);       /* maximal rank of the Cholesky decomposition */
m    = *tempref;
tempref = mxGetPr(prhs[3]);       /* centering */
centering = *tempref;
tempref = mxGetPr(prhs[4]);       /* type of decomposition: 0->Chol, 1->OLS */
kappa = *tempref;
tempref = mxGetPr(prhs[5]);       /* number of candidates performed in advance */
delta = *tempref;
tempref = mxGetPr(prhs[6]);       /*  tolerance  */
tol = *tempref;
if (nrhs>7)
    {
    tempref = mxGetPr(prhs[7]);       /*  display */
    display = *tempref;
    }
else display=0;

if (n<=m+delta) maxcol=n; else maxcol=m+delta;

/* allocate variables */
dJK     = (double*) calloc (n,sizeof(double));
dJY     = (double*) calloc (n,sizeof(double));
dJ      = (double*) calloc (n,sizeof(double));
D       = (double*) calloc (n,sizeof(double));
Dadv    = (double*) calloc (n,sizeof(double));
DK      = (double*) calloc (n,sizeof(double));
error1  = (double*) calloc (m+1,sizeof(double));
error2  = (double*) calloc (m+1,sizeof(double));
G       = (double*) calloc (maxcol*n,sizeof(double));
Gcol    = (double*) calloc (n,sizeof(double));
Gbef    = (double*) calloc (n*(delta+1),sizeof(double));
Gbeftotal = (double*) calloc (n*(delta+1),sizeof(double));
Rbef    = (double*) calloc ((delta+1)*(delta+1),sizeof(double));
Rbeftotal = (double*) calloc ((delta+1)*(delta+1),sizeof(double));
GTGbef  = (double*) calloc ((delta+1)*(delta+1),sizeof(double));
QTYYTQbef = (double*) calloc ((delta+1)*(delta+1),sizeof(double));
GTG     = (double*) calloc (maxcol * maxcol,sizeof(double));
QTYYTQ  = (double*) calloc (maxcol * maxcol,sizeof(double));
meanY   = (double*) calloc (d,sizeof(double));
M2      = (double*) calloc (2*2,sizeof(double));
P       = (int*) calloc (n,sizeof(int));
predicted_gain = (double*) calloc (m+1,sizeof(double));
Q2      = (double*) calloc (2*2,sizeof(double));
QTY     = (double*) calloc (maxcol * d,sizeof(double));
Q       = (double*) calloc (maxcol*n,sizeof(double));
R       = (double*) calloc (maxcol*maxcol,sizeof(double));
R2      = (double*) calloc (2*2,sizeof(double));
tempvec = (double*) calloc (delta+d,sizeof(double));
tempG   = (double*) calloc ((delta+1)*(delta+1),sizeof(double));
tempQ   = (double*) calloc ((delta+1)*(delta+1),sizeof(double));
tempGG  = (double*) calloc ((delta+1),sizeof(double));
tempvec2= (double*) calloc (delta+d,sizeof(double));
true_gain = (double*) calloc (m+1,sizeof(double));
tempA2  = (double*) calloc (d*maxcol,sizeof(double));
tempA3  = (double*) calloc (delta+1,sizeof(double));
tempA4  = (double*) calloc (delta+1,sizeof(double));
A1       = (double*) calloc (n,sizeof(double));
A2       = (double*) calloc (n,sizeof(double));
A3       = (double*) calloc (n,sizeof(double));



/* initialization */
traceK = 0;
for (i=0;i<=n-1;i++) 
    {
    DK[i] = K[i+n*i];
    traceK += DK[i];
    Dadv[i] = DK[i];
    D[i] = DK[i];
    A1[i] = 0.0;
    A2[i] = 0.0;
    A3[i] = 0.0;
    P[i]=i; 
    }
for (i=0;i<=maxcol*n-1;i++) G[i]=0.0;
for (i=0; i<maxcol*maxcol;i++) GTG[i]=0;
for (i=0; i<maxcol*d;i++) QTY[i]=0;

if (centering)
    for (i=0;i<=d-1;i++)
        {
        sumY=0.0;
        nd = n*i;
        for (j=0;j<=n-1;j++) sumY += Y[j+nd];
        meanY[i] = sumY / ( 0.0 + n );
        }
else for (i=0;i<=d-1;i++) meanY[i]=0;
        
sumY2 = 0;
for (i=0;i<=d-1;i++)
    for (j=0;j<=n-1;j++)
         sumY2 += (Y[j+n*i]-meanY[i]) * (Y[j+n*i]-meanY[i]);

mu = (kappa) / sumY2;
lambda = (1.0-kappa) / traceK;
k=0;
kadv=0;
endloop = 1;
force_exit = 0;
error1[0] = traceK;
error2[0] = sumY2;
if (delta>n) delta = n;


/* peforms delta steps of Choleksy + QR */
for (i=0;i<delta;i++)
    {
    kadv++;
    /* select best index */
    diagmax = Dadv[kadv-1];
    jast = kadv-1;
    for(j=kadv-1;j<n;j++) 
        if (Dadv[j]>diagmax / .99)
            {
            diagmax = Dadv[j];
            jast = j;
            } 
    if (diagmax<1e-12)
        {     
        /* cannot add this index, stop the Cholesky decomposition */      
        kadv--;
        i=delta;
        }
    else
        {                                                                                                                                                                                                                                                                     
        /* permute */                                                                                                                                                                                                                                                         
        tempint = P[jast];  P[jast]=P[kadv-1];  P[kadv-1]=tempint;                                                                                                                                                                                                                
        temp = Dadv[jast];  Dadv[jast]=Dadv[kadv-1];  Dadv[kadv-1]=temp;                                                                                                                                                                                          
        temp = D[jast];  D[jast]=D[kadv-1];  D[kadv-1]=temp;                                                                                                                                                                                          
        temp = A1[jast];  A1[jast]=A1[kadv-1];  A1[kadv-1]=temp;                                                                                                                                                                                          
        for (j=0;j<=kadv-2;j++)                                                                                                                                                                                                                                               
            {                                                                                                                                                                                                                                                             
            temp = G[kadv-1+j*n];  G[kadv-1+j*n] = G[jast+j*n];  G[jast+j*n] = temp;                                                                                                                                                                                          
            temp = Q[kadv-1+j*n];  Q[kadv-1+j*n] = Q[jast+j*n];  Q[jast+j*n] = temp;                                                                                                                                                                                          
            }                                                                                                                                                                                                                                                             
                                                                                                                                                                                                                                                                                  
        /* compute new Cholesky column */                                                                                                                                                                                                                                     
        G[kadv-1 + (kadv-1)*n ] = Dadv[kadv-1];                                                                                                                                                                                                                                 
        G[kadv-1 + (kadv-1)*n ] = sqrt(G[kadv-1 + (kadv-1)*n ] );                                                                                                                                                                                                                     
                                                                                                                                                                                                                                                                                  
        for (j=kadv; j<n; j++)                                                                                                                                                                                                                                              
            {                                                                                                                                                                                                                                                                 
            temp=0;                                                                                                                                                                                                                                                           
            for (q=0;q<kadv-1;q++) 
                temp += G[j+n*q] * G[kadv-1 + n*q];                                                                                                                                                                                                            
            G[j+(kadv-1)*n] = ( K[ P[j]+P[kadv-1]*n ] - temp ) / G[kadv-1 + (kadv-1) *n];                                                                                                                                                                                             
            }                                                                                                                                                                                                                                                                 
                                                                                                                                                                                                                                                                                  
        /* update diagonal */                                                                                                                                                                                                                                                 
        for (j=kadv; j<n; j++)                                                                                                                                                                                                                                              
            Dadv[j] -= G[ j + (kadv-1)*n] * G[ j + (kadv-1)*n];                                                                                                                                                                                                               
        Dadv[kadv-1] = 0.0;                                                                                                                                                                                                                                                              
                                                                                                                                                                                                                                                                                  
        /* perform QR decomposition step */                                                                                                                                                                                                                                   
        if (centering)                                                                                                                                                                                                                                                        
            {                                                                                                                                                                                                                                                                 
            temp = 0.0;                                                                                                                                                                                                                                                       
            for (j=0;j<n;j++) 
                temp += G[j+ (kadv-1)*n];                                                                                                                                                                                                                         
            temp = temp / (0.0 + n);                                                                                                                                                                                                                                          
            for (j=0;j<n;j++)
                Gcol[j] = G[j+(kadv-1)*n] - temp;                                                                                                                                                                                                                 
            }                                                                                                                                                                                                                                                                 
        else 
            for (j=0;j<n;j++) 
                Gcol[j] = G[j+(kadv-1)*n];                                                                                                                                                                                                                       
                                                                                                                                                                                                                                                                                  
        for (j=0;j<kadv-1;j++)                                                                                                                                                                                                                                                  
            {                                                                                                                                                                                                                                                                 
            temp = 0.0;                                                                                                                                                                                                                                                         
            for (q=0;q<n;q++) 
                temp += Q[q+j*n] * Gcol[q];                                                                                                                                                                                                                     
            R[j+(kadv-1)*maxcol] = temp;                                                                                                                                                                                                                                        
            }                                                                                                                                                                                                                                                                 
                                                                                                                                                                                                                                                                                  
        for (q=0;q<n;q++)                                                                                                                                                                                                                                                    
            {                                                                                                                                                                                                                                                                 
            temp = 0.0;                                                                                                                                                                                                                                                           
            for (j=0;j<kadv-1;j++) 
                temp += Q[q + j*n] * R[j + (kadv-1)*maxcol];                                                                                                                                                                                                   
            Q[q+(kadv-1)*n] = Gcol[q] - temp;                                                                                                                                                                                                                                   
            }                                                                                                                                                                                                                                                                 
            
        temp = 0.0;                                                                                                                                                                                                                                                             
        for (q=0;q<n;q++) 
            temp+=Q[q + (kadv-1)*n] * Q[q + (kadv-1)*n];                                                                                                                                                                                                            
        R[kadv-1 + (kadv-1)*maxcol] = sqrt(temp);                                                                                                                                                                                                                                 
        for (q=0;q<n;q++) 
            Q[q + (kadv-1)*n] = Q[q + (kadv-1)*n] / R[kadv-1 + (kadv-1)*maxcol];    
            
        /* update cached quantities */
            
        if (centering)                                                                                                                                                                                                                                                        
            for (j=0;j<=kadv-1;j++)    
                for (q=0;q<=n-1;q++)
                    GTG[j+maxcol*(kadv-1)] += G[q + j * n] * G[ q + (kadv-1) * n];    
        else
            for (j=0;j<=kadv-1;j++)    
                for (q=0;q<=kadv-1;q++)
                    GTG[j+maxcol*(kadv-1)] += R[q + j * maxcol] * R[ q + (kadv-1) * maxcol];    
                 
        for (j=0;j<=kadv-1;j++)    
            GTG[kadv-1+maxcol*j] = GTG[j+maxcol*(kadv-1)];
                
        for (j=0;j<d;j++) 
            for (q=0;q<=n-1;q++)
                QTY[kadv-1 + maxcol*j] += Q[q + n*(kadv-1) ] * Y[P[q]+n*j];
            
        for (j=0;j<=kadv-1;j++) 
            {
            QTYYTQ[kadv-1 + maxcol * j] =0.0;
            for (q=0;q<=d-1;q++) 
                QTYYTQ[kadv-1 + maxcol * j] += QTY[kadv-1+ maxcol*q] * QTY[j+ maxcol*q] ;
            QTYYTQ[j + maxcol * (kadv-1) ] = QTYYTQ[kadv-1 + maxcol * j];                     
            }
            
        for (q=kadv-1;q<n;q++)
            {
            A1[q] += GTG[kadv-1 + maxcol* (kadv-1) ] * G[q+(kadv-1)*n] * G[q+(kadv-1)*n] ;
            temp = 0;
            for (j=0;j<=kadv-2;j++)
                temp += G[q+j*n] * GTG[j+maxcol*(kadv-1)];
            A1[q] += 2.0 * temp * G[q+(kadv-1)*n];
            }
            
                                                                                                                                                                          
        }                                                                                                                                                                                                                                                                    
    }
    
   
/* compute remaining costs for all indices */
for (i=0;i<kadv;i++)
    for (j=0;j<d;j++)
        {
        tempA2[i+maxcol*j] = 0.0;
        for (q=0;q<kadv;q++)
            tempA2[i+maxcol*j] += R[q+maxcol*i] * QTY[q+maxcol*j];
        }
for (s=0;s<n;s++)
    for (j=0;j<d;j++)
        {
        temp = 0.0;
        for (i=0;i<kadv;i++)
            temp += G[s+i*n] * tempA2[i+maxcol*j];
        A2[s] += temp * temp;
        }
for (s=0;s<n;s++)
    for (j=0;j<kadv;j++)
        {
        temp = 0.0;
        for (i=0;i<kadv;i++)
            temp += G[s+i*n] * R[j+maxcol*i];
        A3[s] += temp * temp;
        }
        

/* start main loop */
while (endloop & k<m)
    {
    k++;
    
    /* compute approximation scores */                                                                                                                                                                                                                                                                                                                                                                     
    for (i=0;i<=n-k;i++)                                                                                                                                                                                                                                                                                                                                                                                             
        {                                                                                                                                                                                                                                                                                                                                                                                                            
        kast = k+i;                                                                                                                                                                                                                                                                                                                                                                                                
        if (D[kast-1]<1e-12)
            dJK[i] = -1e100;
        else
            {                                                                                                                                                                                                                                                                                                                                                                                                
            dJK[i] = A1[kast-1];                                                                                                                                                                                                                                                                                                                                                                                               
                if (kast > kadv)                                                                                                                                                                                                                                                                                                                                                                                              
                    {                                                                                                                                                                                                                                                                                                                                                                                                        
                    dJK[i] += D[kast-1]*D[kast-1];
                    dJK[i] -= (D[kast-1]-Dadv[kast-1]) * (D[kast-1]-Dadv[kast-1]);                                                                                                                                                                                                                                                                                                                                                                 
                    }                                                                                                                                                                                                                                                                                                                                                                                                        
            dJK[i] = dJK[i] / D[kast-1];                                                                                                                                                                                                                                                                                                                                                                           
            }
        }
    
    /* compute prediction scores */    
    if (kadv>k)                                                                                                                                                                                                                                                                                                                                                                                                        
        {                                                                                                                                                                                                                                                                                                                                                                                                            
        for (i=0;i<=n-k;i++)                                                                                                                                                                                                                                                                                                                                                                                         
            {                                                                                                                                                                                                                                                                                                                                                                                                        
            kast = k+i;   
            if (A3[kast-1] < 1e-12)
                dJY[i]=0.0;
            else 
                dJY[i] = A2[kast-1] / A3[kast-1];                                                                                                                                                                                                                                                                                                                                                                                  
            }                                                                                                                                                                                                                                                                                                                                                                                                        
        }                                                                                                                                                                                                                                                                                                                                                                                                            
    else 
        for (i=0;i<=n-k;i++) 
            dJY[i] = 0;                                                                                                                                                                                                                                                                                                                                                                            
                                                                                                                                                                                                                                                                                                                                                                                                                         
    /* select the best column */                                                                                                                                                                                                                                                                                                                                                                                 
    for (i=0;i<=n-k;i++) 
        dJ[i] = lambda * dJK[i] + mu * dJY[i];   
    diagmax = -1.0;                                                                                                                                                                                                                                                                                                                                                                                                 
    jast = -1;                                                                                                                                                                                                                                                                                                                                                                                                        
    for (j=0;j<=n-k;j++) 
    if (D[j+k-1]>=1e-12) /* only considers the elements that lead to non zero pivot */
        if (dJ[j]> diagmax / 0.9)                                                                                                                                                                                                                                                                                                                                                                          
            {                                                                                                                                                                                                                                                                                                                                                                                
            diagmax = dJ[j];                                                                                                                                                                                                                                                                                                                                                                 
            jast = j;                                                                                                                                                                                                                                                                                                                                                                        
            }
                                        
    if (jast==-1)
        {
        /* no column can be safely added */
        /* simply returns */
        k--;
        endloop=0;
        }      
    else
        {                                                                                                                                                                                                                                                                                                                                                                  
        jast += k;                                                                                                                                                                                                                                                                                                                                                                                                     
        predicted_gain[k-1] = diagmax;            
        if (jast>kadv)
            {
            newpivot = jast;
            jast = kadv + 1;
            }
        else
            {
            a = 1e-12;
            b = -1;
            for (j=kadv;j<n;j++)
                if (Dadv[j]> a / .99)
                    {
                    a = Dadv[j];
                    b = j;
                    }
            if (b==-1)
                newpivot=0;
            else 
                newpivot = b+1;
            }
        
  
        if (newpivot>0)
            {
            kadv++;
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            
            /* permute */                                                                                                                                                                                                                                                         
            tempint = P[newpivot-1];  P[newpivot-1]=P[kadv-1];  P[kadv-1]=tempint;                                                                                                                                                                                                                
            temp = Dadv[newpivot-1];  Dadv[newpivot-1]=Dadv[kadv-1];  Dadv[kadv-1]=temp;                                                                                                                                                                                          
            temp = D[newpivot-1];  D[newpivot-1]=D[kadv-1];  D[kadv-1]=temp;                                                                                                                                                                                          
            temp = A1[newpivot-1];  A1[newpivot-1]=A1[kadv-1];  A1[kadv-1]=temp;                                                                                                                                                                                          
            temp = A2[newpivot-1];  A2[newpivot-1]=A2[kadv-1];  A2[kadv-1]=temp;                                                                                                                                                                                          
            temp = A3[newpivot-1];  A3[newpivot-1]=A3[kadv-1];  A3[kadv-1]=temp;   
            for (j=0;j<=kadv-2;j++)                                                                                                                                                                                                                                               
                {                                                                                                                                                                                                                                                             
                temp = G[kadv-1+j*n];  G[kadv-1+j*n] = G[newpivot-1+j*n];  G[newpivot-1+j*n] = temp;                                                                                                                                                                                          
                temp = Q[kadv-1+j*n];  Q[kadv-1+j*n] = Q[newpivot-1+j*n];  Q[newpivot-1+j*n] = temp;                                                                                                                                                                                          
                }                                                                                                                                                                                                                                                             
                                                                                                                                                                                                                                                                      
            /* compute new Cholesky column */                                                                                                                                                                                                                                     
            G[kadv-1 + (kadv-1)*n ] = Dadv[kadv-1];                                                                                                                                                                                                                                 
            G[kadv-1 + (kadv-1)*n ] = sqrt(G[kadv-1 + (kadv-1)*n ] );                                                                                                                                                                                                                     
                                                                                                                                                                                                                                                                              
            for (j=kadv; j<n; j++)                                                                                                                                                                                                                                              
                {                                                                                                                                                                                                                                                                 
                temp=0;                                                                                                                                                                                                                                                           
                for (q=0;q<kadv-1;q++) temp += G[j+n*q] * G[kadv-1 + n*q];                                                                                                                                                                                                            
                G[j+(kadv-1)*n] = ( K[ P[j]+P[kadv-1]*n ] - temp ) / G[kadv-1 + (kadv-1) *n];                                                                                                                                                                                             
                }                                                                                                                                                                                                                                                                 
                                                                                                                                                                                                                                                                                  
            /* update diagonal */                                                                                                                                                                                                                                                 
            for (j=kadv; j<n; j++)                                                                                                                                                                                                                                              
                Dadv[j] -= G[ j + (kadv-1)*n] * G[ j + (kadv-1)*n];                                                                                                                                                                                                               
            Dadv[kadv-1] = 0.0;                                                                                                                                                                                                                                                              
                                                                                                                                                                                                                                                                                  
            /* perform QR decomposition step */                                                                                                                                                                                                                                   
            if (centering)                                                                                                                                                                                                                                                        
                {                                                                                                                                                                                                                                                                 
                temp = 0.0;                                                                                                                                                                                                                                                       
                for (j=0;j<n;j++) temp += G[j+ (kadv-1)*n];                                                                                                                                                                                                                         
                temp = temp / (0.0 + n);                                                                                                                                                                                                                                          
                for (j=0;j<n;j++) 
                    Gcol[j] = G[j+(kadv-1)*n] - temp;                                                                                                                                                                                                                 
                }                                                                                                                                                                                                                                                                 
            else 
                for (j=0;j<n;j++) 
                    Gcol[j] = G[j+(kadv-1)*n];                                                                                                                                                                                                                       
                                                                                                                                                                                                                                                                                  
            for (j=0;j<kadv-1;j++)                                                                                                                                                                                                                                                  
                {                                                                                                                                                                                                                                                                 
                temp = 0;                                                                                                                                                                                                                                                         
                for (q=0;q<n;q++) 
                    temp += Q[q+j*n] * Gcol[q];                                                                                                                                                                                                                     
                R[j+(kadv-1)*maxcol] = temp;                                                                                                                                                                                                                                        
                }                                                                                                                                                                                                                                                                 
                                                                                                                                                                                                                                                                                  
             for (q=0;q<n;q++)                                                                                                                                                                                                                                                    
                {                                                                                                                                                                                                                                                                 
                temp=0;                                                                                                                                                                                                                                                           
                for (j=0;j<kadv-1;j++) 
                    temp += Q[q + j*n] * R[j + (kadv-1)*maxcol];                                                                                                                                                                                                   
                Q[q+(kadv-1)*n] = Gcol[q] - temp;                                                                                                                                                                                                                                   
                }                                                                                                                                                                                                                                                                 
            temp = 0;                                                                                                                                                                                                                                                             
            for (q=0;q<n;q++) 
                temp+=Q[q + (kadv-1)*n] * Q[q + (kadv-1)*n];                                                                                                                                                                                                            
            R[kadv-1 + (kadv-1)*maxcol] = sqrt(temp);                                                                                                                                                                                                                                 
            for (q=0;q<n;q++) 
                Q[q + (kadv-1)*n] = Q[q + (kadv-1)*n] / R[kadv-1 + (kadv-1)*maxcol];    
            
            /* update cached quantities */
            
            if (centering)                                                                                                                                                                                                                                                        
                {
                for (j=k-1;j<=kadv-1;j++)    
                    for (q=0;q<=n-1;q++)
                        GTG[j+maxcol*(kadv-1)] += G[q + j * n] * G[ q + (kadv-1) * n];    
                }
            else
                 {
                for (j=k-1;j<=kadv-1;j++)    
                    for (q=0;q<=kadv-1;q++)
                        GTG[j+maxcol*(kadv-1)] += R[q + j * maxcol] * R[ q + (kadv-1) * maxcol];    
                }   
            for (j=k-1;j<=kadv-1;j++)    
                GTG[kadv-1+maxcol*j] = GTG[j+maxcol*(kadv-1)];
                
            for (j=0;j<d;j++) 
                for (q=0;q<=n-1;q++)
                    QTY[kadv-1 + maxcol*j] += Q[q + n*(kadv-1) ] * Y[P[q]+n*j];
            
            for (j=k-1;j<=kadv-1;j++) 
                {
                QTYYTQ[kadv-1 + maxcol * j] =0.0;
                for (q=0;q<=d-1;q++) 
                    QTYYTQ[kadv-1 + maxcol * j] += QTY[kadv-1+ maxcol*q] * QTY[j+ maxcol*q] ;
                QTYYTQ[j + maxcol * (kadv-1) ] = QTYYTQ[kadv-1 + maxcol * j];                     
                }
                       
            
            for (q=kadv-1;q<n;q++)
                {
                A1[q] += GTG[kadv-1 + maxcol* (kadv-1) ] * G[q+(kadv-1)*n] * G[q+(kadv-1)*n] ;
                temp = 0;
                for (j=k-1;j<=kadv-2;j++)
                    temp += G[q+j*n] * GTG[j+maxcol*(kadv-1)];
                A1[q] += 2.0 * temp * G[q+(kadv-1)*n];               
                }
            
            temp = 0.0;
            for (q=k-1;q<kadv;q++)  
                temp+= R[q+maxcol*(kadv-1)]*R[q+maxcol*(kadv-1)];
            for (q=kadv-1;q<n;q++)
                A3[q] += G[q+(kadv-1)*n] * G[q+(kadv-1)*n] * temp;
            
            for (q=k-1;q<=kadv-2;q++)
                {
                tempA3[q-k+1]=0.0;
                for (j=k-1;j<=kadv-1;j++)
                    tempA3[q-k+1] += R[j+maxcol*(kadv-1)] * R[j+maxcol*q];
                }
            for (q=kadv-1;q<n;q++)
                for (j=k-1;j<=kadv-2;j++)
                    A3[q] += 2.0 * G[q+(kadv-1)*n] * G[q+j*n] * tempA3[j-k+1];
                
            
            temp =0.0;
            for (q=k-1;q<=kadv-1;q++)
                {
                tempA3[q-k+1]=0.0;
                for (j=k-1;j<=kadv-1;j++)
                    tempA3[q-k+1] += R[j+maxcol*(kadv-1)] * QTYYTQ[j+maxcol*q];
                temp += tempA3[q-k+1] * R[q+maxcol*(kadv-1)];
                }
            for (q=kadv-1;q<n;q++)
                A2[q] += G[q+(kadv-1)*n] * G[q+(kadv-1)*n] * temp;
                
            for (q=k-1;q<=kadv-2;q++)
                {
                tempA4[q-k+1] = 0.0;
                for (j=k-1;j<=kadv-1;j++)
                    tempA4[q-k+1] += tempA3[j-k+1] * R[j+maxcol*q];
                }
                
            for (q=kadv-1;q<n;q++)
                for (j=k-1;j<=kadv-2;j++)
                    A2[q] += 2.0 * G[q+(kadv-1)*n] * G[q+j*n] * tempA4[j-k+1];
                    
            }
                                                                                                                                                                                                                                                                                                                                                                                                                   
                                                                                                                                                                                                                                                                           
                                                                                                                                                                                                                                                                                                                                                                                                                         
        /* permute indices p and q in the decompositions */
            
        p = k;                                                                                                                                                                                                                                                                                                                                                                                                     
        q = jast;
        if (q>p)                                                                                                                                                                                                                                                                                                                                                                                                     
            {
            /* store */
            for (i=0;i<=q-p;i++)
                for (j=0;j<n;j++)
                    Gbef[j+n*i] = G[j+n*(i+p-1)];

            for (i=0;i<=kadv-k;i++)
                for (j=0;j<n;j++)
                    Gbeftotal[j+n*i] = G[j+n*(i+k-1)];
                    
            for (i=0;i<=q-p;i++)
                for (j=0;j<=q-p;j++)
                    GTGbef[j+(q-p+1)*i] = GTG[j+p-1+maxcol*(i+p-1)];                                                                                                                                                                                                                                                                                                                                                                                                            
 
            for (i=0;i<=q-p;i++)
                for (j=0;j<=kadv-k;j++)
                    QTYYTQbef[i+(q-p+1)*j] = QTYYTQ[j+k-1+maxcol*(i+k-1)];                                                                                                                                                                                                                                                                                                                                                                                                            
            
            for (i=0;i<=q-p;i++)
                for (j=0;j<=q-p;j++)
                    Rbef[j+(q-p+1)*i] = R[j+p-1+maxcol*(i+p-1)];                                                                                                                                                                                                                                                                                                                                                                                                            
 
            for (i=0;i<=kadv-k;i++)
                for (j=0;j<=kadv-k;j++)
                    Rbeftotal[j+(delta+1)*i] = R[j+k-1+maxcol*(i+k-1)];                                                                                                                                                                                                                                                                                                                                                                                                            
            
            for (i=0;i<=q-p;i++)
                for (j=0;j<=q-p;j++)
                    if (i==j) 
                        tempG[i+(q-p+1)*j]=1.0;
                    else 
                        tempG[i+(q-p+1)*j]=0.0;
             
            for (i=0;i<=q-p;i++)
                for (j=0;j<=q-p;j++)
                    if (i==j) 
                        tempQ[i+(q-p+1)*j]=1.0;
                    else 
                        tempQ[i+(q-p+1)*j]=0.0;
                    
                    
            /* permute succesive indices in the decomposition */
            for (s=q-1;s>=p;s--)
                {
                /* permute */                                                                                                                                                                                                                                                         
                tempint = P[s-1];  P[s-1]=P[s];  P[s]=tempint;                                                                                                                                                                                                                
                temp = Dadv[s-1];  Dadv[s-1]=Dadv[s];  Dadv[s]=temp;                                                                                                                                                                                          
                temp = D[s-1];  D[s-1]=D[s];  D[s]=temp;                                                                                                                                                                                          
                temp = A1[s-1];  A1[s-1]=A1[s];  A1[s]=temp;
                temp = A2[s-1];  A2[s-1]=A2[s];  A2[s]=temp;
                temp = A3[s-1];  A3[s-1]=A3[s];  A3[s]=temp;                                                                                                                                                                                          
                for (j=0;j<=kadv-1;j++)                                                                                                                                                                                                                                               
                    {                                                                                                                                                                                                                                                             
                    temp = G[s+j*n];  G[s+j*n] = G[s-1+j*n];  G[s-1+j*n] = temp;                                                                                                                                                                                          
                    temp = Q[s+j*n];  Q[s+j*n] = Q[s-1+j*n];  Q[s-1+j*n] = temp;                                                                                                                                                                                          
                    }
                for (j=0;j<=q-p;j++)                                                                                                                                                                                                                                               
                    {
                    temp = Gbef[s+j*n];  Gbef[s+j*n] = Gbef[s-1+j*n];  Gbef[s-1+j*n] = temp;      
                    }                                                                                                                                                                                    
                for (j=0;j<=kadv-k;j++)                                                                                                                                                                                                                                               
                    {
                    temp = Gbeftotal[s+j*n];  Gbeftotal[s+j*n] = Gbeftotal[s-1+j*n];  Gbeftotal[s-1+j*n] = temp;                                                                                                                                                                                          
                    }
                                                                                                                                                                                                                                                                                 
                /* perform permutation of indices in Cholesky and QR */
                M2[0+2*0]=G[s-1+n*(s-1)]; M2[0+2*1]=G[s+n*(s-1)];
                M2[1+2*0]=G[s-1+n*s];     M2[1+2*1]=G[s+n*s];
                qr2(M2,Q2,R2);
                for (j=0;j<n;j++)
                    {
                    temp2 = G[j+n*s];
                    temp1 = G[j+n*(s-1)];
                    G[j+n*(s-1)] = temp1 * Q2[0+2*0] + temp2 * Q2[1+2*0];
                    G[j+n*s]     = temp1 * Q2[0+2*1] + temp2 * Q2[1+2*1];
                    }
                G[s-1+n*s]=0.0;
                for (j=0;j<kadv;j++)
                    {
                    temp1 = R[j+maxcol*(s-1)];
                    temp2 = R[j+maxcol*s];                  
                    R[j+maxcol*(s-1)] = temp1 * Q2[0+2*0] + temp2 * Q2[1+2*0];
                    R[j+maxcol*s]     = temp1 * Q2[0+2*1] + temp2 * Q2[1+2*1];
                    }
                for (j=k-1;j<=kadv-1;j++)
                    {
                    temp1 = GTG[j+maxcol*(s-1)];
                    temp2 = GTG[j+maxcol*(s)];                   
                    GTG[j+maxcol*(s-1)] =  temp1 * Q2[0+2*0] + temp2 * Q2[1+2*0];
                    GTG[j+maxcol*s]     = temp1 * Q2[0+2*1] + temp2 * Q2[1+2*1];
                    }
                for (j=k-1;j<=kadv-1;j++)
                    {
                    temp1 = GTG[s-1+maxcol*j];
                    temp2 = GTG[s+maxcol*j];              
                    GTG[s-1+maxcol*j] = temp1 * Q2[0+2*0] + temp2 * Q2[1+2*0];
                    GTG[s+maxcol*j]     = temp1 * Q2[0+2*1] + temp2 * Q2[1+2*1];
                    }
                for (j=0;j<=(q-p);j++)
                    {
                    temp1 = tempG[j+(q-p+1)*(s-p)];                    
                    temp2 = tempG[j+(q-p+1)*(s-p+1)];
                    tempG[j+(q-p+1)*(s-p)]       = temp1 * Q2[0+2*0] + temp2 * Q2[1+2*0];
                    tempG[j+(q-p+1)*(s-p+1)]     = temp1 * Q2[0+2*1] + temp2 * Q2[1+2*1];
                     }  
                    
                M2[0+2*0]=R[s-1+maxcol*(s-1)]; M2[0+2*1]=R[s-1+maxcol*s];
                M2[1+2*0]=R[s+maxcol*(s-1)];   M2[1+2*1]=R[s+maxcol*s];
                qr2(M2,Q2,R2);
                for (j=0;j<n;j++)
                    {
                    temp1 = Q[j+n*(s-1)];
                    temp2 = Q[j+n*(s)];
                    Q[j+n*(s-1)] =  temp1 * Q2[0+2*0] + temp2 * Q2[1+2*0];
                    Q[j+n*s]     = temp1 * Q2[0+2*1] + temp2 * Q2[1+2*1];
                    }
                for (j=0;j<kadv;j++)
                    {
                    temp1 = R[s-1+maxcol*j];
                    temp2 = R[s+maxcol*j];
                    R[s-1+maxcol*j] = temp1 * Q2[0+2*0] + temp2 * Q2[1+2*0];
                    R[s+maxcol*j]   = temp1 * Q2[0+2*1] + temp2 * Q2[1+2*1];
                    }
                for (j=0;j<d;j++)
                    {
                    temp1 = QTY[s-1+maxcol*j];
                    temp2 = QTY[s+maxcol*j];
                    
                    QTY[s-1+maxcol*j] = temp1 * Q2[0+2*0] + temp2 * Q2[1+2*0];
                    QTY[s+maxcol*j]   = temp1 * Q2[0+2*1] + temp2 * Q2[1+2*1];
                    }           
                 R[s+maxcol*(s-1)]=0.0;
                for (j=k-1;j<=kadv-1;j++)
                    {
                    temp1 = QTYYTQ[j+maxcol*(s-1)];
                    temp2 = QTYYTQ[j+maxcol*(s)];
                    QTYYTQ[j+maxcol*(s-1)] =  temp1 * Q2[0+2*0] + temp2 * Q2[1+2*0];
                    QTYYTQ[j+maxcol*s]     = temp1 * Q2[0+2*1] + temp2 * Q2[1+2*1];
                    }
                for (j=k-1;j<=kadv-1;j++)
                    {
                    temp1 = QTYYTQ[s-1+maxcol*j];
                    temp2 = QTYYTQ[s+maxcol*j];
                    QTYYTQ[s-1+maxcol*j] = temp1 * Q2[0+2*0] + temp2 * Q2[1+2*0];
                    QTYYTQ[s+maxcol*j]     = temp1 * Q2[0+2*1] + temp2 * Q2[1+2*1];
                    }
                    
                    
                for (j=0;j<=(q-p);j++)
                    {
                    temp1 = tempQ[j+(q-p+1)*(s-p)];                    
                    temp2 = tempQ[j+(q-p+1)*(s-p+1)];
                    tempQ[j+(q-p+1)*(s-p)]       = temp1 * Q2[0+2*0] + temp2 * Q2[1+2*0];
                    tempQ[j+(q-p+1)*(s-p+1)]     = temp1 * Q2[0+2*1] + temp2 * Q2[1+2*1];
                    }
                }    
                                                                                                                                                                                                                                                                                                                                                                                   
            /* update cached values */
            for (i=0;i<=q-p;i++) 
                tempGG[i]=0.0;
            for (i=0;i<=q-p;i++)
                for (j=0;j<=q-p;j++)
                    tempGG[i] += GTGbef[i+(q-p+1)*j] * tempG[j];                  
                    
            for (i=k-1;i<n;i++)
                for (j=0;j<=q-p;j++)
                    A1[i] -= 2.0 * G[i+n*(k-1)] * Gbef[i+n*j] * tempGG[j];
                    
                    
            for (i=k-1;i<n;i++)
                for (j=k-1;j<p-1;j++)
                    A1[i] -= 2.0 * G[i+n*(k-1)] * G[i+n*j] * GTG[j+maxcol*(k-1)];
                    
            for (i=k-1;i<n;i++)
                for (j=q;j<kadv;j++)
                    A1[i] -= 2.0 * G[i+n*(k-1)] * G[i+n*j] * GTG[j+maxcol*(k-1)];
            
            for (j=0;j<=kadv-k;j++)
                { 
                tempA3[j]=0.0;
                for (i=0;i<=q-p;i++)
                    tempA3[j]+= tempQ[i] * QTYYTQbef[i+(q-p+1)*j];
                }
            
            for (j=0;j<=kadv-k;j++)
                { 
                tempA4[j]=0.0;
                for (i=0;i<=kadv-k;i++)
                    tempA4[j]+= tempA3[i] * Rbeftotal[i+(delta+1)*j];
                }
      
            for (j=0;j<=q-p;j++)
                {
                tempA3[j]=0.0;
                for (i=0;i<=q-p;i++)
                    tempA3[j]+= Rbef[i+(q-p+1)*j] * tempQ[i];
                }
                
            for (i=k-1;i<n;i++)
                { 
                temp = 0.0;
                for (j=k-1;j<p-1;j++)
                    temp += G[i+n*j] * R[(k-1)+maxcol*j];
                for (j=q;j<kadv;j++)
                    temp += G[i+n*j] * R[(k-1)+maxcol*j];
                for (j=0;j<=q-p;j++)
                    temp += Gbef[i+n*j] * tempA3[j];
                A3[i] -= temp * temp;
                A2[i] += temp * temp * QTYYTQ[k-1+maxcol*(k-1)];
                       
                for (j=0;j<=kadv-k;j++)
                    A2[i] -= 2.0 * temp * Gbeftotal[i+n*j] * tempA4[j];                     
                }
            }
        else
            {
            /* no need to permute, still update some */
            for (i=k-1;i<n;i++)
                for (j=k-1;j<kadv;j++)
                    A1[i] -= 2.0 * G[i+n*(k-1)] * G[i+n*j] * GTG[j+maxcol*(k-1)];
               
            for (j=0;j<=kadv-k;j++)
                { 
                tempA4[j]=0.0;
                for (i=0;i<=kadv-k;i++)
                    tempA4[j]+= QTYYTQ[k-1+maxcol*(i+k-1)] * R[i+k-1+(maxcol)*(j+k-1)];
                }
               
            for (i=k-1;i<n;i++)
                { 
                temp = 0.0;
                for (j=k-1;j<kadv;j++)
                    temp += G[i+n*j] * R[(k-1)+maxcol*j];
                A3[i] -= temp * temp;
                A2[i] += temp * temp * QTYYTQ[k-1+maxcol*(k-1)];
                for (j=0;j<=kadv-k;j++)
                    A2[i] -= 2.0 * temp * G[i+n*(j+k-1)] * tempA4[j];
                }
            }


        /* update diagonal */                                                                                                                                                                                                                                                 
        for (j=k; j<n; j++)                                                                                                                                                                                                                                              
            D[j] -= G[ j + (k-1)*n] * G[ j + (k-1)*n];                                                                                                                                                                                                               
        D[k-1] = 0.0;    
            
        /* update costs */
        for (i=k-1;i<n;i++)
            A1[i] += GTG[k-1 + maxcol* (k-1) ] * G[i+(k-1)*n] * G[i+(k-1)*n] ;
               

    
        /* compute errors */
        temp2 = 0.0;                                                                                                                                                                                                                                                                                                                                                                                                   
        for (s=0;s<d;s++)                                                                                                                                                                                                                                                                                                                                                                                            
            {   
            temp = 0.0;                                                                                                                                                                                                                                                                                                                                                                                                 
            for (i=0;i<n;i++) 
                temp += Q[i+n*(k-1)] * ( Y[P[i]+n*s] - meanY[s]);                                                                                                                                                                                                                                                                                                                                                    
            temp2 += temp * temp;                                                                                                                                                                                                                                                                                                                                                                                    
            }   
                                                                                                                                                                                                                                                                                                                                                                                                              
        temp1 = 0.0;                                                                                                                                                                                                                                                                                                                                                                                                   
        for (i=0;i<n;i++) 
            temp1 += G[i+n*(k-1)] * G[i+n*(k-1)];                                                                                                                                                                                                                                                                                                                                                       
        true_gain[k-1] = temp1 * lambda + temp2 * mu;   
        error1[k] = error1[k-1]-temp1;                                                                                                                                                                                                                                                                                                                                                                               
        error2[k] = error2[k-1]-temp2;  
        if (true_gain[k-1] < tol)                                                                                                                                                                                                                                                                                                                                                                                    
            endloop=0;                                                                                                                                                                                                                                                                                                                                                                                               
        }                                                                                                                                                                                                                                                                                                                                                                                                          
    }

/* prepare output */
plhs[0] = mxCreateDoubleMatrix(n, k, 0);      /* G */
plhs[1] = mxCreateDoubleMatrix(1, n, 0);      /* P */
plhs[2] = mxCreateDoubleMatrix(n, k, 0);      /* Q */
plhs[3] = mxCreateDoubleMatrix(k, k, 0);      /* R */
plhs[4] = mxCreateDoubleMatrix(1, k+1, 0);    /* error1 */
plhs[5] = mxCreateDoubleMatrix(1, k+1, 0);    /* error2 */
plhs[6] = mxCreateDoubleMatrix(1, k+1, 0);    /* error */
plhs[7] = mxCreateDoubleMatrix(1, k, 0);      /* predicted_gain */
plhs[8] = mxCreateDoubleMatrix(1, k, 0);      /* true_gain */

                                                                                                                                                                                                                                                                                                                                                                                        

y = mxGetPr(plhs[0]); 
for (i=0;i<n*k;i++) 
    y[i]=G[i];
y = mxGetPr(plhs[1]);
for (i=0;i<n;i++) 
    y[i]=(double) P[i]+1;
y = mxGetPr(plhs[2]);
for (i=0;i<n*k;i++) 
    y[i]=Q[i];
y = mxGetPr(plhs[3]);
for (i=0;i<k;i++) 
    for (j=0;j<k;j++) 
        y[i+k*j]=R[i+maxcol*j];
y = mxGetPr(plhs[4]);      
for (i=0;i<=k;i++) 
    y[i]=error1[i] /traceK;
y = mxGetPr(plhs[5]);
for (i=0;i<=k;i++) 
    y[i]=error2[i]/sumY2;
y = mxGetPr(plhs[6]);
for (i=0;i<=k;i++) 
    y[i]=lambda * error1[i] + mu *error2[i];
y = mxGetPr(plhs[7]);      
for (i=0;i<k;i++) 
    y[i]=predicted_gain[i];
y = mxGetPr(plhs[8]);
for (i=0;i<k;i++) 
    y[i]=true_gain[i];


 
/* free variables */

free(dJK);
free(dJY);
free(dJ);
free(Dadv);
free(D);
free(DK);
free(error1);
free(error2);
free(G);
free(Gcol);
free(M2);
free(meanY);  
free(predicted_gain);
free(P);
free(QTY);
free(Q2);
free(Q);
free(R);
free(R2);
free(GTG);
free(QTYYTQ);
free(QTYYTQbef);
free(Gbef);
free(Gbeftotal);
free(Rbef);
free(Rbeftotal);
free(GTGbef);
free(tempA2);
free(tempA3);
free(tempA4);
free(tempvec);
free(tempvec2);
free(true_gain);
free(tempG);
free(tempQ);
free(tempGG);
free(A1);
free(A2);
free(A3);
             
return;
}
