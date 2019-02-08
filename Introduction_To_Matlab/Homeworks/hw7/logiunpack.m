function L = logiunpack(cv)
    n = length(cv);
    L = false(n);
    for ii = 1:n
        for jj = 1:length(cv{ii})
            L(ii,cv{ii}(jj)) = true;
        end
    end
end