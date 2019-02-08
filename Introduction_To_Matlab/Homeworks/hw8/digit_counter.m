function n = digit_counter(fname)
    n = -1;
    fid = fopen(fname,'r');
    if fid >= 0
        n = sum(isstrprop(fread(fid,inf,'char=>char'),'digit'));
        fclose(fid);
    end
end