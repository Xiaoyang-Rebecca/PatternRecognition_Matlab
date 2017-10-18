function Dist = fun_FeaDist(X, Y, stype)

if strcmp(stype,'LBP')
    Dist = fun_FeaLBPDist(X, Y);
else
    Dist = EuDist2(X', Y',0);
end

