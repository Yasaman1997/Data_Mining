function ok = eligible(v,q)
    ok = mean([v q]) >= 92 && v > 88 && q > 88;
end