function h = halfsum(A)
    % there is a built-in MATLAB function for almost anything...

    h = sum(sum(triu(flipud(A))))
end