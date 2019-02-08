function s = spiral_diag_sum(n)
%   The last element is always n square and it is in the corner.
%   A simple arithmetic expression gives the sum of the four corners.
%   Do this for a series of odd numbers from 3 to n.
%   Add 1 for the starting value in the center

    v = 3:2:n;
    s = 1 + sum(4*v.^2 - 6*(v-1)); 
end