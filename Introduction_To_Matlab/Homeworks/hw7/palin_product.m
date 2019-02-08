function n = palin_product (dig,lim)

% a is the smallest dig-digit number that can be formed. If the smallest possible
% product (a^2) is smaller than the specified limit, we determine b, the largest
% dig-digit number that can be formed. We then build the square outer product of a:b.
% Logically indexing into to this matrix for elements less than lim creates a column
% vector P of candidate products. We convert each of these to a string, reverse its
% characters, and convert it back to a number, to form the column vector Q. Finally,
% we return the maximum element in P which has the same value in both P and Q.

    n = 0;
    a = 10^(dig-1);
    if lim>a^2
        b = 10^dig - 1;
        P = (a:b)' * (a:b);
        P = P(P<lim);
        Q = str2num(fliplr(num2str(P)));
        n = max(P(P==Q));
    end 
end