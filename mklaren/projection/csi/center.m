function M=center(M);
M = M - 1/size(M,1) * repmat(sum(M,1),size(M,1),1);
