function cl = integerize(A)
    cls = {'int8'; 'int16'; 'int32'; 'int64'};
    cl = 'NONE';
    mx = max(A(:));
    mn = min(A(:));
    for ii = 1:length(cls)
        if intmax(cls{ii}) >= mx && intmin(cls{ii}) <= mn
            cl = cls{ii};
            break;
        end
    end
end