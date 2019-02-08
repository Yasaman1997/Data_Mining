function Q = intquad(n,m)
    Q = zeros(n,m);
    Q = [Q Q+1; Q+2 Q+3];
end