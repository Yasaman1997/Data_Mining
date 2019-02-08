function D = reverse_diag(n)
    D = zeros(n);
    D(n:max([1,n-1]):max([n,n^2-1])) = 1;
end