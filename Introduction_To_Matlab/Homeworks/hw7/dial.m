function n = dial (s)
    if ~all(ismember(s,['0':'9','A':'Z'])) || length(s) > 16
        n = uint64(0);
    else
        map = '22233344455566677778889999';
        s(s>='A') = map(s(s>='A')-64);
        n = uint64(str2double(s)) ;
    end
end  