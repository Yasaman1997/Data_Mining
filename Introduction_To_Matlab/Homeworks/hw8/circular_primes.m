function n = circular_primes(k)
    n = 0;
    for p = primes(k-1)
        if circular_prime(p)
            n = n + 1;
        end
    end
end

function yes = circular_prime(k)
    d = num2str(k);
    for ii = 1:length(d)
        d = circshift(d,[0 1]);   % try next circular shifted version
        if ~isprime(str2num(d))   
            yes = false;          % not circular prime
            return;               % so we can return
        end
    end
    yes = true;                   % if we made it here, it is a circular prime
end