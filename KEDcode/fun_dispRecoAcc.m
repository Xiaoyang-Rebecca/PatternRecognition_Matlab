function reco_ratio = fun_dispRecoAcc(TestClass,tt_ID,inds,remark)

% disp recognition Accuracy for each subset

% Reference:
% K. K. Huang, D. Q. Dai, C. X. Ren and Z. R. Lai, Learning Kernel Extended Dictionary for Face Recognition,
% IEEE Transation on Neural Network Learning System, vol. pp, no. pp, 1-13, 2016
% Email: kkcocoon@163.com (K. K. Huang)

if length(inds)==0
    inds{1} = 1:length(TestClass);
end

if ~exist('remark')
    remark = '';
end

reco_ratio= [];
for i=1:length(inds)
    ind = inds{i};
    tr = (sum(tt_ID(ind)==TestClass(ind)))/length(ind);
    tr = round(10000*tr)/100;
    reco_ratio = [reco_ratio,tr];
end
disp([fun_Format(reco_ratio), '% ', remark]);
% disp([num2str(reco_ratio)]); 