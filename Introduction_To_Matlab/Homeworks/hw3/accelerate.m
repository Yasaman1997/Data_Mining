function a = accelerate(F1,F2,m)
    F = F1 + F2;
    f = sqrt(sum(F.^2));
    a = f/m;
end