function [Q,R] = qr2(M);
% QR decomposition for 2x2 matrices (this is to make sure that the C and
% Matlab implementations output exactly the same matrices
Q = zeros(2,2);
R = zeros(2,2);
x = sqrt( M(1,1)^2 + M(2,1)^2 );
R = x;
Q(:,1) = M(:,1) / x;
R(1,2) = Q(:,1)' * M(:,2);
Q(:,2) = M(:,2) - R(1,2) * Q(:,1);
R(2,2) = norm(Q(:,2));
Q(:,2) = Q(:,2) / R(2,2);
