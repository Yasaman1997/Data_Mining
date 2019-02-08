function month = year2016(m)
    if ~isscalar(m) || m < 1 || m > 12 || m ~= floor(m)
        month = [];
        return;
    end
    days = ([31 29 31 30 31 30 31 31 30 31 30 31]);
    ms = {'January'; 'February'; 'March'; 'April'; 'May'; 'June'; ...
           'July'; 'August'; 'September'; 'October'; 'November'; 'December'}; 
    ds = {'Sun'; 'Mon'; 'Tue'; 'Wed'; 'Thu'; 'Fri'; 'Sat'};

    start = 4;  % Jan 1, 2016 was a Friday. US week starts on Sunday.
                % We'll add ii and 1 below because rem(n,7) returns numbers
                % 0-6 and we need indexes 1-7. 
    start = start + sum(days(1:m-1));
    for ii = 1:days(m)
        month(ii).month = ms{m};
        month(ii).date = ii;
        month(ii).day = ds{rem(start+ii,7)+1};
    end
end