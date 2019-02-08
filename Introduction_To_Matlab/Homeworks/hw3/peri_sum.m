function s = peri_sum(A)
    B = A(2:end-1,2:end-1);
    s = sum(A(:)) - sum(B(:));
end