%function [out] = centuries(n)
%    if ~isscalar(n) || n<1 || n>3000 || n~=floor(n)
%        out = '';
%    else
%        cents = {'I'; 'II'; 'III'; 'IV'; 'V'; 'VI'; 'VII'; 'VIII'; 'IX'; 'X';
%              'XI'; 'XII'; 'XIII'; 'XIV'; 'XV'; 'XVI'; 'XVII'; 'XVIII'; 'XIX'; 'XX';
%              'XXI'; 'XXII'; 'XXIII'; 'XXIV'; 'XXV'; 'XXVI'; 'XXVII'; 'XXVIII'; 'XXIX'; 'XXX'};
%        out = cents{ceil(n/100)};
%    end
%end

function c = centuries (y)
    c = '';
    if isscalar(y) && rem(y,1)==0 && y>0 && y<=3000
        c = A2R(fix((y-1)/100)+1);
    end
end

function R = A2R (A)
    % Converts Arabic numbers to Roman strings.
    Roman  = {'I' 'IV' 'V' 'IX' 'X' 'XL' 'L' 'XC' 'C' 'CD' 'D' 'CM' 'M'};
    Arabic = {1 4 5 9 10 40 50 90 100 400 500 900 1000};
    R = ''; k = 13;
    while k>0                    % remove largest modulii first
        if A>=Arabic{k}          % if value>current modulus
            A = A-Arabic{k};     %   remove modulus from value
            R = [R Roman{k}];    %   append Roman character
        else
            k = k-1;             % else consider next smaller modulus
        end
    end
end