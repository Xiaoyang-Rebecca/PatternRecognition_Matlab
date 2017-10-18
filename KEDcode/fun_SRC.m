function [reco_ratio,tt_ID,tsrc] = fun_SRC(tTrainX,TrainClass,tTestX,TestClass)

% Reference:
% K. K. Huang, D. Q. Dai, C. X. Ren and Z. R. Lai, Learning Kernel Extended Dictionary for Face Recognition,
% IEEE Transation on Neural Network Learning System, vol. pp, no. pp, 1-13, 2016
% Email: kkcocoon@163.com (K. K. Huang)

TrainClass = TrainClass(1:size(tTrainX,2));

classids   =    unique(TrainClass);
NumClass   =    length(classids);
tr_num     =    size(tTrainX,2);
tt_num     =    size(tTestX,2);
tt_ID      =    zeros(1,tt_num);
gap        =    zeros(1,NumClass);

tTrainX  =  tTrainX./ repmat(sqrt(sum(tTrainX.*tTrainX)),[size(tTrainX,1) 1]); 
tTestX   =  tTestX./ repmat(sqrt(sum(tTestX.*tTestX)),[size(tTestX,1) 1]); 

bt=clock;
ds=0;
for i  =  1:tt_num  
    s = SolveDALM(tTrainX, tTestX(:,i),'lambda',0.001);

    for j   =  1:NumClass
        ind = find(TrainClass==classids(j));
        temp_s =  zeros(size(s));
        temp_s(ind)  =  s(ind);
        zz     =  tTestX(:,i)-tTrainX*temp_s;
        gap(j) =  zz(:)'*zz(:); 
    end
        
    [mg,mi] = min(gap);
    tt_ID(i)  =  classids(mi);

    if  ds==0 & etime(clock,bt)>1800
        ds=i;
    end
    if ds>0 & mod(i,ds)==0
        reco_ratio=(sum(tt_ID(1:i)==TestClass(1:i)))/i; reco_ratio = round(10000*reco_ratio)/100;
        tt = clock;   
        disp(['% SRC testing: ' num2str(i) '/' num2str(tt_num) '. reco_ratio=' num2str(reco_ratio) ', Elapsed time: ' num2str(round(etime(clock,bt)/60)) ' min. time=' num2str(round(tt(4))) ':' num2str(round(tt(5)))]);
    end
end
% toc;

reco_ratio=(sum(tt_ID==TestClass(1:tt_num)))/tt_num; % output the recognition rate
reco_ratio = round(10000*reco_ratio)/100;
tsrc = etime(clock,bt);
tsrc = num2str(round(tsrc));