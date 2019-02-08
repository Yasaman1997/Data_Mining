function [mx ind] = max_product(v,n)
    ind = -1;
    if isempty(v) || n > length(v)
        mx = 0;
        return;
    end
    mx = -Inf;
    for ii = 1:length(v)-n+1
        mul = prod(v(ii:ii+n-1));
        if mul > mx
            mx = mul;
            ind = ii;
        end
    end
end



