function [a,k] = approximate_e (delta)
    e = exp(1);                      % reference value for e
    f = 1;                           % first factorial term  
    a = 1;                           % first series term
    k = 0;                           % term subscript
    while abs(a-e)>delta
        k = k+1;    
        f = f*k;
        a = a+1/f;
    end
end