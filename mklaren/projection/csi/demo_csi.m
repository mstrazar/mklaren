% demonstration script for the csi decomposition algorithm
clear all

% create two-class toy data
n = 500;
m = 100;

n = n * 2;
x1 = [ ...
    2 * [ 1 -.5 ; -.5 1 ] * randn(2,n/4)  + repmat([ -3; 5], 1, n/4) , ...
    [ 1 -.5 ; -.5 1 ] * randn(2,n/4)  + repmat([ 3; -.5], 1, n/4 ) ];
x2 = [ ...
    [ 1 -.5 ; -.5 1 ] * randn(2,n/4)  + repmat([ 2; 3], 1, n/4) ,...
    [ 1 -.5 ; -.5 1 ] * randn(2,n/4)  + repmat([ 6; 0 ], 1, n/4 ) ];
x = [ x1 x2 ];
y = [ ones(1,n/2),  zeros(1,n/2) ;   zeros(1,n/2),  ones(1,n/2) ];
rp = randperm(n);
xtest = x(:,rp(n/2:end));
ytest = y(:,rp(n/2:end));
x = x(:,rp(1:n/2));
y = y(:,rp(1:n/2));
subplot(2,2,1);
plot(x(1,find(y(1,:)==1)),x(2,find(y(1,:)==1)),'bx'); hold on
plot(x(1,find(y(2,:)==1)),x(2,find(y(2,:)==1)),'rx'); hold off
title('data');

% LS SVM parameters
alpha = .5;
kappa = 1e-4;

% CSI decomposition with two look ahead parameters
tic
[G,P,Q,R,error1,error2] = csi_gaussian(x,alpha,y',m,1,.99,40,1e-8);
csi_time_40=toc

tic
[GL,PL,QL,RL,error1L,error2L] = csi_gaussian(x,alpha,y',m,1,.99,80,1e-8);
csi_time_80=toc

% regular Cholesky without look_ahead
tic
[GC,PC,QC,RC,error1C,error2C] = csi_gaussian(x,alpha,y',m,1,0,0,1e-8);
csi_no_lookahead_time=toc

subplot(2,2,2);
plot(error1,'b'); hold on;
plot(error1L,'g'); hold on;
plot(error1C,'r'); hold off;
legend('CSI-40','CSI-80','Cholesky');
title('approximation of kernel matrix');


subplot(2,2,3);
plot(error2,'b'); hold on;
plot(error2L,'g'); hold on;
plot(error2C,'r'); hold off;
legend('CSI-40','CSI-80','Cholesky');
title('Prediction of side information (training data)');


% compte test set accuracies for all steps of the decomposition
% using an LS SVM
n = size(G,1);
ntest = size(ytest,2);
k = size(G,2);
L = exp( - alpha * sqdist(xtest,x(:,P(1:k))));
testing_errors=zeros(k,1);
Minv=[];
for i=1:k
    r = R(1:i,i);
    Minv = [ Minv zeros(i-1,1); zeros(1,i-1) 1/n/kappa];
    Minv = Minv - ( Minv * r ) * ( r' * Minv ) * inv( 1 + r' * Minv * r ) ;
    Gtalpha = R(1:i,1:i)' * ( Minv * ( Q(:,1:i)' * y(:,P)' ) );
    b =  - mean( G(:,1:i) * ( Gtalpha ) ) + mean(y');
    beta = ( G(1:i,1:i)' ) \ Gtalpha;
    Zhat = L(:, 1:i )*beta + ones(ntest,1)*b;
    distances=zeros(ntest,2);
    for j=1:2
        delta=zeros(1,2); delta(j)=1;
        distances(:,j)= sum( ( Zhat - repmat(delta,ntest,1) ).^2 ,2 );
    end
    [a,Zpred] = min(distances,[],2);
    Ztest = ytest(2,:)+1;
    testing_errors(i)=length(find(Ztest-Zpred'~=0))/ntest;
end


% compte test set accuracies for all steps of the decomposition
% using an LS SVM
n = size(GL,1);
ntest = size(ytest,2);
k = size(GL,2);
L = exp( - alpha * sqdist(xtest,x(:,PL(1:k))));
testing_errorsL=zeros(k,1);
Minv=[];
for i=1:k
    r = RL(1:i,i);
    Minv = [ Minv zeros(i-1,1); zeros(1,i-1) 1/n/kappa];
    Minv = Minv - ( Minv * r ) * ( r' * Minv ) * inv( 1 + r' * Minv * r ) ;
    Gtalpha = RL(1:i,1:i)' * ( Minv * ( QL(:,1:i)' * y(:,PL)' ) );
    b =  - mean( GL(:,1:i) * ( Gtalpha ) ) + mean(y');
    beta = ( GL(1:i,1:i)' ) \ Gtalpha;
    Zhat = L(:, 1:i )*beta + ones(ntest,1)*b;
    distances=zeros(ntest,2);
    for j=1:2
        delta=zeros(1,2); delta(j)=1;
        distances(:,j)= sum( ( Zhat - repmat(delta,ntest,1) ).^2 ,2 );
    end
    [a,Zpred] = min(distances,[],2);
    Ztest = ytest(2,:)+1;
    testing_errorsL(i)=length(find(Ztest-Zpred'~=0))/ntest;
end


% compte test set accuracies for all steps of the decomposition
% using an LS SVM for the Cholesky decompostion with no look ahead
n = size(GC,1);
ntest = size(ytest,2);
k = size(GC,2);
L = exp( - alpha * sqdist(xtest,x(:,PC(1:k))));
testing_errorsC=zeros(k,1);
Minv=[];
for i=1:k
    r = RC(1:i,i);
    Minv = [ Minv zeros(i-1,1); zeros(1,i-1) 1/n/kappa];
    Minv = Minv - ( Minv * r ) * ( r' * Minv ) * inv( 1 + r' * Minv * r ) ;
    Gtalpha = RC(1:i,1:i)' * ( Minv * ( QC(:,1:i)' * y(:,PC)' ) );
    b =  - mean( GC(:,1:i) * ( Gtalpha ) ) + mean(y');
    beta = ( GC(1:i,1:i)' ) \ Gtalpha;
    Zhat = L(:, 1:i )*beta + ones(ntest,1)*b;
    distances=zeros(ntest,2);
    for j=1:2
        delta=zeros(1,2); delta(j)=1;
        distances(:,j)= sum( ( Zhat - repmat(delta,ntest,1) ).^2 ,2 );
    end
    [a,Zpred] = min(distances,[],2);
    Ztest = ytest(2,:)+1;
    testing_errorsC(i)=length(find(Ztest-Zpred'~=0))/ntest;
end



subplot(2,2,4);
plot(testing_errors,'b'); hold on
plot(testing_errorsL,'g'); hold on
plot(testing_errorsC,'r'); hold off
legend('CSI-40','CSI-80','Cholesky');
title('Prediction of side information (testing data)');
