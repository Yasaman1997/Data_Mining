function [s avg] = sindeg(d)
    s = sin(d*pi/180);
    avg = mean(s(:));
end