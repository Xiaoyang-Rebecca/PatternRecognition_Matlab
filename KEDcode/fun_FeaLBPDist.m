function Dist = fun_FeaLBPDist(X, Y)

% Function: get the X2 distance between two rotation-invariant LBP mappings;
% If you use this code, please cite the following paper.

% Reference:
% K. K. Huang, D. Q. Dai, C. X. Ren and Z. R. Lai, Learning Kernel Extended Dictionary for Face Recognition,
% IEEE Transation on Neural Network Learning System, vol. pp, no. pp, 1-13, 2016
% Email: kkcocoon@163.com (K. K. Huang)

n1 = size(X,2);
n2 = size(Y,2);
Dist = zeros(n1,n2);
for i = 1:n1
    for j= 1:n2
        Dist(i,j) = sum((X(:,i)-Y(:,j)).^2 ./ (0.001+X(:,i)+Y(:,j)));  % X2 kernel
    end
end