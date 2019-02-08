function m = number2letters (n)

% A is a 3x10 array of the letter counts of the words used to form various numbers, with
% units in row 1, "teens" in row 2 and tens in row 3. For the supplied number, we first
% determine the number of units, tens and hundreds. These three values determine the
% words and thus the letter count to be returned.
 
  A = [ 0 3 3 5 4 4 3 5 5 4; ...              % units
        3 6 6 8 8 7 7 9 8 8; ...              % "teens"
        0 0 6 6 5 5 5 7 6 6];                 % tens
  h = fix(n/100); 
  t = fix(rem(n,100)/10); 
  u = rem(n,10);
  if h>0,     m = A(1,h+1)+7;                % h 'hundred'
  else        m = 0; 
  end
  switch t
    case 0,   m = m+A(1,u+1);                % units only
    case 1,   m = m+A(2,u+1);                % "teens" only
    otherwise m = m+A(3,t+1)+A(1,u+1);       % tens and units
  end 
end