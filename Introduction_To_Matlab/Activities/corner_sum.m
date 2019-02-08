function s = corner_sum(A)
[m,n]=size(A);
s=A(1,1)+A(1,n)+A(m,1)+A(m,n);