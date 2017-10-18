function [reco_ratio,tt_ID,tt_ID_sample,tsrc] = fun_ESRC(QTrainX,TrainClass,TestX,TestClass)

% Reference:
% K. K. Huang, D. Q. Dai, C. X. Ren and Z. R. Lai, Learning Kernel Extended Dictionary for Face Recognition,
% IEEE Transation on Neural Network Learning System, vol. pp, no. pp, 1-13, 2016
% Email: kkcocoon@163.com (K. K. Huang)

classids   =    unique(TrainClass);
NumClass   =    length(classids);
tt_num     =    size(TestX,2);
tt_ID      =    zeros(1,tt_num);
gap        =    zeros(1,NumClass);

all_num     =    size(QTrainX,2);
tr_num   =    length(TrainClass);

bt = clock; ds=0;
QTrainX  =  QTrainX./ repmat(sqrt(sum(QTrainX.*QTrainX)),[size(QTrainX,1) 1]); 
TestX   =  TestX./ repmat(sqrt(sum(TestX.*TestX)),[size(TestX,1) 1]); 

tt_ID = zeros(1,tt_num);
for i  =  1:tt_num 
    y = TestX(:,i);
    s = SolveDALM(QTrainX,y,'lambda',0.001);
    for j = 1:NumClass
        ind = find(TrainClass == classids(j));
        ind = [ind, tr_num+1:all_num];
        temp_s =  zeros(size(s));
        temp_s(ind)  =  s(ind);
        zz     =  y  - QTrainX*temp_s;
        gap(j) =  zz(:)'*zz(:); 
    end 
    [mg,mi] = min(gap);
    tt_ID(i)  = classids(mi);    
    [~,tt_ID_sample(i)] = max(abs(s)); 
    
    if  ds==0 & etime(clock,bt)>600 %1800
        ds=i;
    end
    if ds>0 & mod(i,ds)==0
        reco_ratio=(sum(tt_ID(1:i)==TestClass(1:i)))/i; reco_ratio = round(10000*reco_ratio)/100;
        tt = clock;   
        disp(['% ESRC testing: ' num2str(i) '/' num2str(tt_num) '. reco_ratio=' num2str(reco_ratio) ', Elapsed time: ' num2str(round(etime(clock,bt)/60)) ' min. time=' num2str(round(tt(4))) ':' num2str(round(tt(5)))]);
    end    
end

tsrc = etime(clock,bt);
tsrc = num2str(round(tsrc));

reco_ratio=(sum(tt_ID==TestClass(1:tt_num)))/tt_num; % output the recognition rate
reco_ratio = round(10000*reco_ratio)/100;
% disp(['ESRC:', num2str(reco_ratio)]);
% disp(['% ESRC time: ' num2str(round(cputime-bt)) ]);

