function fea = fun_FeaLBP_block(X,opt)

% multi-scale block-wise LBP feature.
% If you use this code, please cite the following paper.

% Reference:
% K. K. Huang, D. Q. Dai, C. X. Ren and Z. R. Lai, Learning Kernel Extended Dictionary for Face Recognition,
% IEEE Transation on Neural Network Learning System, vol. pp, no. pp, 1-13, 2016
% Email: kkcocoon@163.com (K. K. Huang)


if ~exist('opt') opt=[]; end
if ~isfield(opt,'MAPPING')  opt.MAPPING=getmapping(8,'u2'); end
if ~isfield(opt,'blknum_h') opt.blknum_h=14; end
if ~isfield(opt,'blknum_w') opt.blknum_w=12; end
if ~isfield(opt,'lbp_radius') opt.lbp_radius=2; end
if ~isfield(opt,'num_scales') opt.num_scales=3; end
if ~isfield(opt,'scale_factor') opt.scale_factor=2^(1/2); end  

MAPPING = opt.MAPPING;
lbp_radius = opt.lbp_radius;

scale = 1;
fea = [];
for si = 1:opt.num_scales
    tX = imresize(X,1/scale,'bilinear');
    blknum_h = round(opt.blknum_h/scale);
    blknum_w = round(opt.blknum_w/scale); if si==5 blknum_w=2; end
    
%     figure(si); imshow(tX,'border','tight','initialmagnification','fit'); hold on;  
%     Xg = tX;

    Xg = fun_FeaLBP(tX,lbp_radius,8,MAPPING,'i'); % max(X(:))=58;
    
    [im_h,im_w]=size(Xg);
    hn = blknum_h; 
    
    feas = [];
    hi=1;
    while hn>0
        bh = round((im_h-hi+1) / hn);  
        he = hi+bh-1;    

%         plot([1 im_w+5],[he,he], 'y-','linewidth',4);

        wi=1; 
        wn = blknum_w;
        while wn>0
            bw = floor((im_w-wi+1) / wn); 
            we = wi+bw-1;
            
%             plot([we we],[1,im_h+5], 'y-','linewidth',4);
            
            img = Xg(hi:he,wi:we);
            imglbp = hist(img(:),0:58);
            feas = [feas; imglbp(:)];
            
            wn = wn -1;
            wi = wi + bw;
        end
        
        hn = hn -1;
        hi = hi + bh;
    end
    
%     set(gcf,'Position',[100,200,4*size(tX,1),4*size(tX,2)]);
%     axis normal;  
%     f=getframe(gcf); imwrite(f.cdata,['fig_Grid' num2str(si) '.png']);

    fea = [fea; feas];
    scale = scale * opt.scale_factor;
end

