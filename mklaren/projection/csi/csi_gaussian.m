function [G,P,Q,R,error1,error2,error,predicted_gain,true_gain] = csi_gaussian(x,alpha,Y,m,centering,kappa,delta,tol)
% INPUT
% x  : input data nvar x n
% alpha : kernel parameter of the RBF kernel
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

K = exp( - alpha * sqdist(x,x) );
[G,P,Q,R,error1,error2,error,predicted_gain,true_gain] = csi(K,centering,kappa,delta,tol)



