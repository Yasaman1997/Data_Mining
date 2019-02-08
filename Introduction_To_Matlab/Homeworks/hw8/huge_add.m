function summa = huge_add(a,b)
    if ~ischar(a) || ~ischar(b) || sum(isstrprop(a,'digit')) ~= length(a) || ...
            sum(isstrprop(b,'digit')) ~= length(b)
        summa = -1;
        return;
    end
    lng = max([length(a) length(b)]);
    a = [a(end:-1:1) '0'+zeros(1,lng-length(a))];   % flip and pad with zeros if necessary
    b = [b(end:-1:1) '0'+zeros(1,lng-length(b))];   % flip and pad with zeros if necessary
    carry = 0;
    for ii = 1:lng
        c = carry + str2num(a(ii)) + str2num(b(ii)) % add the digits plus the caryy
        carry = c > 9;                              % is there carry? (0 or 1)
        summa(ii) = num2str(mod(c,10));             % put the character in the result
    end
    if carry
        summa(end+1) = '1';                         % need a leading 1 if carry
    end
    summa = summa(end:-1:1);                        % flip it back
end