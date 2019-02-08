function mul = smallest_multiple(n)
    facts = zeros(1,n);             % store the exponents of various factors
    for ii = 2:n
        f = factor(ii);             % get factors for current integer
        for jj = 2:ii
            k = sum(f == jj);       % what's the exponent of this factor?
            if k > facts(jj)        % if it is greater than what we have so far
                facts(jj) = k;      % update to this new value
            end
        end
    end
    % Compute the result with one command. 
    % The 'native' option tells MATLAB to work in uint64
    mul = prod(uint64((1:n).^facts),'native');   
    if mul == intmax('uint64')
       mul = uint64(0);
    end
end


%function r = smallest_multiple(k) 
%    r = uint64(1);
%    for n = 1:k
%        r = r * (n / gcd(r,n));   % Include factor not already present in r
%    end
%    if r == intmax('uint64')
%       r = uint64(0);
%    end
%end