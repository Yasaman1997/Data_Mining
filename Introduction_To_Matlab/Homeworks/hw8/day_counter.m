function n = day_counter(year)
    months = [31 28 31 30 31 30 31 31 30 31 30 31];     % dyas of the months
    start = 1776;                                       
    y = year - start;                                   % years since 1776
    % number of days since January 1, 1776:
    ndays = y*365 + ceil(y/4) - floor((year-1)/100) + floor(start/100) + floor((year-1)/2000);
    % if leap year, need to adjust February:
    if mod(year,4) == 0 && (mod(year,100) ~= 0 || mod(year,400) == 0)
        months(2) = 29;
    end
    months = [0 cumsum(months(1:end-1))];               % number of days in the previous months
    n = sum(mod(ndays+months,7) == 0);                  % mod() == 0 means Monday
end