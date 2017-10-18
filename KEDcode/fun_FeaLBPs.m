function [Gs]=fun_FeaLBPs(Is,opt)

% Extract LBP features from image set 'Is', a column for each sample.
% If you use this code, please cite the following paper.

% Reference:
% K. K. Huang, D. Q. Dai, C. X. Ren and Z. R. Lai, Learning Kernel Extended Dictionary for Face Recognition,
% IEEE Transation on Neural Network Learning System, vol. pp, no. pp, 1-13, 2016
% Email: kkcocoon@163.com (K. K. Huang)

if ~exist('opt') opt=[]; end

if ~isfield(opt,'MAPPING') 
    opt.MAPPING=getmapping(8,'u2'); 
end

if length(size(Is))==2
    if isfield(opt,'im_h') & isfield(opt,'im_w')
        im_h = opt.im_h;
        im_w = opt.im_w;
        Is = reshape(Is,[im_h,im_w,size(Is,2)]);
    else
        error('Please specify opt.im_h and opt.im_w.');
    end
end

if length(size(Is))~=3
    error('Wrong input data.');
end

Gs = [];
for n=1:size(Is,3)
    LBPHIST = fun_FeaLBP_block(Is(:,:,n),opt);
    if n==1
        Gs = zeros(length(LBPHIST),size(Is,3));
    end
    Gs(:,n) = LBPHIST(:);
end
