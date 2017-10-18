%% ------------------------------------------------------------------
% KED matlab code on CAS-PEAL dataset, by K. K. Huang.
% Please download CAS-PEAL dataset first: https://yunpan.cn/crC8N5ZNfGbiw, password:0c39, size: 89M
%
% Reference:
% K. K. Huang, D. Q. Dai, C. X. Ren and Z. R. Lai, Learning Kernel Extended Dictionary for Face Recognition,
% IEEE Transation on Neural Network Learning System, vol. pp, no. pp, 1-13, 2016
% Email: kkcocoon@163.com (K. K. Huang)
%
% To create the training set, we randomly select some subjects in three kinds
% of subsets, i.e., 200 subjects from lighting subset, 100 from expression
% subset and 20 from accessory subset, with 4 samples for each subject. Besides
% the 1280 samples, we also add the normal sample of a subject if any sample
% of the subject appears in the training set. To form the gallery set, we use
% all the normal images, i.e., 1040 images of the 1040 subjects, with each 
% subject having only one image. Then we create six probe sets correspond to 
% the six subsets: expression, lighting, accessory, background, distance, and time. 
% All the images that appear in the training set are excluded from these probe sets.
% The training information is saved by 'db_PEAL_ind_tr_1.mat'.
%% ------------------------------------------------------------------

clc;clear;
warning off;

feaType = 'LBP';
tt=clock;
%% ------------------------------------------------------------------------
% load original face images
% The CAS-PEAL database is cropped by K. K. Huang, 
% which can be download from: https://yunpan.cn/crC8N5ZNfGbiw, password:0c39, size: 89M
% if you use the data, please cite our TNNLS2016 paper.
load('db_PEAL.mat');  
ind_gal=ind0_gal;ind_gal=ind0_gal; ind_dis=ind0_dis; ind_bac=ind0_bac; ind_age=ind0_age;

% Extract LBP feature if not exists.
sLBPFile = 'db_PEAL_LBP.mat';
dirLBPFile = dir(sLBPFile);
if size(dirLBPFile,1)==0
    opt=[]; opt.blknum_h=14; opt.blknum_w=12; opt.num_scales=3; opt.im_h=im_h;  opt.im_w=im_w; 
    TrainXg = fun_FeaLBPs(TrainX,opt);
    save(sLBPFile,'TrainXg','TrainClass');
end

load('db_PEAL_LBP'); % load LBP feature
load(['db_PEAL_ind_tr_1']);  % training information: ligting: 200*4; expression: 100*4; accessary:20*4

%% ------------------------------------------------------------------------
% Training samples, including the occluded samples
Xtr = TrainXg(:,[ind_tr,ind_tr_acc]);
XtrClass = TrainClass([ind_tr,ind_tr_acc]);

% Gallery samples
Xgal = TrainXg(:,ind_gal);
XgalClass = TrainClass(ind_gal);

% Testing samples
disp('% Accessory, Lighting, Expression, Time, Background, Distance, Glasses, Sunglass, Hat');
Xtest = TrainXg(:,[ind_acc,ind_lig,ind_exp,ind_age,ind_bac,ind_dis]);
XtestClass = TrainClass([ind_acc,ind_lig,ind_exp,ind_age,ind_bac,ind_dis]);
inds = [];
inds{1} = 1:length(ind_acc);
inds{end+1} = length(ind_acc)+1:length([ind_acc,ind_lig]);
inds{end+1} = length([ind_acc,ind_lig])+1:length([ind_acc,ind_lig,ind_exp]);
inds{end+1} = length([ind_acc,ind_lig,ind_exp])+1:length([ind_acc,ind_lig,ind_exp,ind_age]);
inds{end+1} = length([ind_acc,ind_lig,ind_exp,ind_age])+1:length([ind_acc,ind_lig,ind_exp,ind_age,ind_bac]);
inds{end+1} = length([ind_acc,ind_lig,ind_exp,ind_age,ind_bac])+1:length(XtestClass);

% Index for each kind of subset.
inds_acc12=[];inds_acc3=[];inds_acc4=[];
imageList_acc = imageList([ind_acc]);
for i=1:length(imageList_acc)
    s = imageList_acc{i};
    ind = strfind(s,'_A');
    if str2num(s(ind+2))>=4 
        inds_acc4 = [inds_acc4,i];
    elseif str2num(s(ind+2))==3 
        inds_acc3 = [inds_acc3,i];
    elseif str2num(s(ind+2))>=1 
        inds_acc12 = [inds_acc12,i];
    end
end
inds{end+1}=inds_acc12; 
inds{end+1}=inds_acc3; 
inds{end+1}=inds_acc4; 

%% ------------------------------------------------------------------------
NumTrain = length(XtrClass);
classids = unique(XtrClass);
NumClass = length(classids);

%% PCA
PCADim = 600;
W_pca  =  Eigenface_f(Xtr,PCADim);

Xtr_pca   =  W_pca'* Xtr;
Xgal_pca    =  W_pca'* Xgal;
Xtest_pca    =  W_pca'* Xtest;

[reco_ratio,PCA_ID] = fun_NN(Xgal_pca,XgalClass,Xtest_pca,XtestClass);
fun_dispRecoAcc(XtestClass,XgalClass(PCA_ID),inds,['PCA' num2str(PCADim) '_NN'] );

%% KDA --------------------------------------------------------------  
% The training process for KDA may take long time. 
DistTrain = fun_FeaDist(Xtr,Xtr,feaType);
DistTest_gal  = fun_FeaDist(Xtr,Xgal,feaType);
DistTest_test  = fun_FeaDist(Xtr,Xtest,feaType);

sigma = 3 * sum(sqrt(DistTrain(:))) / NumTrain^2;
Ktrain = exp(-DistTrain./sigma^2);
Ktest_gal  = exp(-DistTest_gal./sigma^2);
Ktest_test  = exp(-DistTest_test./sigma^2);

options = [];   
options.Kernel = 1; 
W_kda = KDA(options,XtrClass,Ktrain);  

pTest_tr    =  W_kda' * Ktrain;
pTest_gal    =  W_kda' * Ktest_gal;
pTest_test    =  W_kda' * Ktest_test;

[reco_ratio,KDA_ID] = fun_NN(pTest_gal,XgalClass,pTest_test,XtestClass);
fun_dispRecoAcc(XtestClass,XgalClass(KDA_ID),inds,['KDA'] );

[reco_ratio,KDA_SRC_ID,tsrc] = fun_SRC(pTest_gal,XgalClass,pTest_test,XtestClass);
fun_dispRecoAcc(XtestClass,KDA_SRC_ID,inds,['KDA_SRC, t' tsrc]);

%% KED --------------------------------------------------
% PairOcc: Occluded training samples index, from file 'db_PEAL_ind_tr_1.mat'.
NumOcc = 10;
pTrainOcc=[];
OccType =[0 1];
for i=1:length(OccType)
    ind = find(PairOcc(:,3)==OccType(i));
    tPairOcc = PairOcc(ind,:);
    Tro = TrainXg(:,tPairOcc(:,1)); % occluded sample
    Trc = TrainXg(:,tPairOcc(:,2)); % corresponding normal sample
    tOcc = fun_KED_dict(Xtr,Tro,Trc,W_kda,sigma,feaType,NumOcc); % The key function
    pTrainOcc = [pTrainOcc,tOcc];
end

[~,KDE_ID,~,tsrc] = fun_ESRC([pTest_gal,pTrainOcc],XgalClass,pTest_test,XtestClass);
fun_dispRecoAcc(XtestClass,KDE_ID,inds,['KED(' num2str(length(OccType)) ')' num2str(size(pTest_gal,2)) '+' num2str(size(pTrainOcc,2)) ', t' tsrc]);

