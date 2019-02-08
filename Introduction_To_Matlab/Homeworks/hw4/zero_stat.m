function pct = zero_stat(A)
    pct = 100 * (numel(A) - sum(A(:))) / numel(A);
end