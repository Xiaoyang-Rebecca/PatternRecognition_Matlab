% this funciton is design to find best dimension for feature reduction and
% classifier combination using adhoc.
clear all

addpath('.\Results')
addpath('.\SVM')
addpath('.\GMM')
load('data.mat')
X_Train = data.Xtrain';  %750*NoF 750 samples,NoF features
X_Test  = data.Xval';

global Y_Train
Y_Train = data.Ytrain;  %750*1   750 samples,target=1..15 (15 classes)
global Y_Test_Desired
Y_Test_Desired  = data.Yval;   %the desire output of target(class value)

[NoF,NoTr]=size(X_Train);
NoC=max(Y_Train);


% %------------------------------------------------------------------------
% %------Generate Dimension reducted data for all dimensions 
% FeatureReductor='LDA';  %Dim<15
% for Dim= 1:(NoC-1);   
%    [X_Train_Proj,X_Test_Proj]=FeatureReduction(X_Train,Y_Train,X_Test,FeatureReductor,Dim);
%    ProjData(Dim).ProjDim=Dim;
%    ProjData(Dim).X_Train_Proj=X_Train_Proj;
%    ProjData(Dim).X_Test_Proj=X_Test_Proj;
% end   
% ProjectedData(1)=struct('FeatureReductor',FeatureReductor,'ProjData',ProjData);
% clear ProjData
% 
% FeatureReductor='PCA';  
% for Dim= 1:NoF;   
%    [X_Train_Proj,X_Test_Proj]=FeatureReduction(X_Train,Y_Train,X_Test,FeatureReductor,Dim);
%    ProjData(Dim).ProjDim=Dim;
%    ProjData(Dim).X_Train_Proj=X_Train_Proj;
%    ProjData(Dim).X_Test_Proj=X_Test_Proj;
% end   
% ProjectedData(2)=struct('FeatureReductor',FeatureReductor,'ProjData',ProjData);
% clear ProjData
% 
% FeatureReductor='KLDA';  %Dim<15
% for Dim= 1:(NoC-1);   
%    [X_Train_Proj,X_Test_Proj]=FeatureReduction(X_Train,Y_Train,X_Test,FeatureReductor,Dim);
%    ProjData(Dim).ProjDim=Dim;
%    ProjData(Dim).X_Train_Proj=X_Train_Proj;
%    ProjData(Dim).X_Test_Proj=X_Test_Proj;
% end   
% ProjectedData(3)=struct('FeatureReductor',FeatureReductor,'ProjData',ProjData);
% clear ProjData
% 
% FeatureReductor='KPCA';  
% for Dim= 1:NoF;   
%    [X_Train_Proj,X_Test_Proj]=FeatureReduction(X_Train,Y_Train,X_Test,FeatureReductor,Dim);
%    ProjData(Dim).ProjDim=Dim;
%    ProjData(Dim).X_Train_Proj=X_Train_Proj;
%    ProjData(Dim).X_Test_Proj=X_Test_Proj;
% end   
% ProjectedData(4)=struct('FeatureReductor',FeatureReductor,'ProjData',ProjData);
% clear ProjData
% 
% FeatureReductor='NONE';  
%    ProjData(1).ProjDim=NoF;
%    ProjData(1).X_Train_Proj=X_Train';
%    ProjData(1).X_Test_Proj=X_Test';
% ProjectedData(5)=struct('FeatureReductor',FeatureReductor,'ProjData',ProjData);
% clear ProjData
% 
% save -v7.3 '.\Results\ProjectedData.mat' ProjectedData;
%storage the feature reducted data set into Results folder


load('.\Results\ProjectedData.mat')

load('.\Results\AcaPlot.mat')

%%
%--------------------------------------------------------------------------
%-------------------------------------------------------------------------
% %find best Parameter for All Feature Reductor and classifier combination


% %% ------------------GassianML  classify------------------------
% Classifier='GaussianML';
% % Adhoc Plot for LDA--GassianML  classify
% DimM=(NoC-1);  %exam dimension from 1 to DimM
% FeatureReductor='LDA';
% Accuracy=AdhocAnalysis(ProjectedData,Y_Train,Y_Test_Desired,FeatureReductor,DimM,Classifier);
% AcaPlot(1)=struct('FeatureReductor',FeatureReductor,'Classifier',Classifier,'Accuracy',Accuracy);
%  
% % Adhoc Plot for PCA--GassianML  classify
% DimM=50;  %exam dimension from 1 to DimM
% FeatureReductor='PCA';
% Accuracy=AdhocAnalysis(ProjectedData,Y_Train,Y_Test_Desired,FeatureReductor,DimM,Classifier);
% AcaPlot(2)=struct('FeatureReductor',FeatureReductor,'Classifier',Classifier,'Accuracy',Accuracy);
% 
% % Adhoc Plot for KLDA--GassianML  classify
% DimM=(NoC-1);  %exam dimension from 1 to DimM
% FeatureReductor='KLDA';
% Accuracy=AdhocAnalysis(ProjectedData,Y_Train,Y_Test_Desired,FeatureReductor,DimM,Classifier);
% AcaPlot(3)=struct('FeatureReductor',FeatureReductor,'Classifier',Classifier,'Accuracy',Accuracy);
% 
% % Adhoc Plot for KPCA--GassianML  classify
% DimM=50;  %exam dimension from 1 to DimM
% FeatureReductor='KPCA';
% Accuracy=AdhocAnalysis(ProjectedData,Y_Train,Y_Test_Desired,FeatureReductor,DimM,Classifier);
% AcaPlot(4)=struct('FeatureReductor',FeatureReductor,'Classifier',Classifier,'Accuracy',Accuracy);
% 
% %% 
% %%------------------KNN  classify
% Classifier='KNN';KMax=10;
% % Adhoc Plot for LDA--KNN  classify
% DimM=(NoC-1);  %exam dimension from 1 to DimM
% FeatureReductor='LDA';
% Accuracy=AdhocAnalysis(ProjectedData,Y_Train,Y_Test_Desired,FeatureReductor,DimM,Classifier,KMax);
% AcaPlot(5)=struct('FeatureReductor',FeatureReductor,'Classifier',Classifier,'Accuracy',Accuracy);
%  
% %Adhoc Plot for PCA--KNN  classify
% DimM=50;  %exam dimension from 1 to DimM
% FeatureReductor='PCA';
% Accuracy=AdhocAnalysis(ProjectedData,Y_Train,Y_Test_Desired,FeatureReductor,DimM,Classifier,KMax);
% AcaPlot(6)=struct('FeatureReductor',FeatureReductor,'Classifier',Classifier,'Accuracy',Accuracy);
% 
% % Adhoc Plot for KLDA--KNN  classify
% DimM=(NoC-1);  %exam dimension from 1 to DimM
% FeatureReductor='KLDA';
% Accuracy=AdhocAnalysis(ProjectedData,Y_Train,Y_Test_Desired,FeatureReductor,DimM,Classifier,KMax);
% AcaPlot(7)=struct('FeatureReductor',FeatureReductor,'Classifier',Classifier,'Accuracy',Accuracy);
% 
% %Adhoc Plot for KPCA--KNN  classify
% DimM=50;  %exam dimension from 1 to DimM
% FeatureReductor='KPCA';
% Accuracy=AdhocAnalysis(ProjectedData,Y_Train,Y_Test_Desired,FeatureReductor,DimM,Classifier,KMax);
% AcaPlot(8)=struct('FeatureReductor',FeatureReductor,'Classifier',Classifier,'Accuracy',Accuracy);
warning('off','all')
% 
%%------------------GMM  classify
Classifier='GMM';KMax=5;DimM=14;  %exam dimension from 1 to DimM
% Adhoc Plot for LDA--GMM  classify

FeatureReductor='LDA';
Accuracy=AdhocAnalysis(ProjectedData,Y_Train,Y_Test_Desired,FeatureReductor,DimM,Classifier,KMax);
AcaPlot(9)=struct('FeatureReductor',FeatureReductor,'Classifier',Classifier,'Accuracy',Accuracy);
  save -v7.3 '.\Results\AcaPlot.mat' AcaPlot;
%Adhoc Plot for PCA--GMM  classify
%DimM=14;  %exam dimension from 1 to DimM
FeatureReductor='PCA';
Accuracy=AdhocAnalysis(ProjectedData,Y_Train,Y_Test_Desired,FeatureReductor,DimM,Classifier,KMax);
AcaPlot(10)=struct('FeatureReductor',FeatureReductor,'Classifier',Classifier,'Accuracy',Accuracy);
 save -v7.3 '.\Results\AcaPlot.mat' AcaPlot;
% Adhoc Plot for KLDA--GMM  classify
%DimM=14;  %exam dimension from 1 to DimM
FeatureReductor='KLDA';
Accuracy=AdhocAnalysis(ProjectedData,Y_Train,Y_Test_Desired,FeatureReductor,DimM,Classifier,KMax);
AcaPlot(11)=struct('FeatureReductor',FeatureReductor,'Classifier',Classifier,'Accuracy',Accuracy);
 save -v7.3 '.\Results\AcaPlot.mat' AcaPlot;
%Adhoc Plot for KPCA--GMM  classify
%DimM=14;  %exam dimension from 1 to DimM
FeatureReductor='KPCA';
Accuracy=AdhocAnalysis(ProjectedData,Y_Train,Y_Test_Desired,FeatureReductor,DimM,Classifier,KMax);
AcaPlot(12)=struct('FeatureReductor',FeatureReductor,'Classifier',Classifier,'Accuracy',Accuracy);
 save -v7.3 '.\Results\AcaPlot.mat' AcaPlot;
%% 
%------------------KSVM  classify

Classifier='KSVM'; %kernel SVM (Kernel function rbf)

% Adhoc Plot for LDA--KSVM  classify
DimM=(NoC-1);  %exam dimension from 1 to DimM
FeatureReductor='LDA';
Accuracy=AdhocAnalysis(ProjectedData,Y_Train,Y_Test_Desired,FeatureReductor,DimM,Classifier);
AcaPlot(13)=struct('FeatureReductor',FeatureReductor,'Classifier',Classifier,'Accuracy',Accuracy);
 
% Adhoc Plot for PCA--KSVM  classify
DimM=50;  %exam dimension from 1 to DimM
FeatureReductor='PCA'; 
Accuracy=AdhocAnalysis(ProjectedData,Y_Train,Y_Test_Desired,FeatureReductor,DimM,Classifier);
AcaPlot(14)=struct('FeatureReductor',FeatureReductor,'Classifier',Classifier,'Accuracy',Accuracy);

% Adhoc Plot for KLDA--KSVM  classify
DimM=(NoC-1);  %exam dimension from 1 to DimM
FeatureReductor='KLDA';
Accuracy=AdhocAnalysis(ProjectedData,Y_Train,Y_Test_Desired,FeatureReductor,DimM,Classifier);
AcaPlot(15)=struct('FeatureReductor',FeatureReductor,'Classifier',Classifier,'Accuracy',Accuracy);

% Adhoc Plot for KPCA--KSVM  classify
DimM=50;  %exam dimension from 1 to DimM
FeatureReductor='KPCA';
Accuracy=AdhocAnalysis(ProjectedData,Y_Train,Y_Test_Desired,FeatureReductor,DimM,Classifier);
AcaPlot(16)=struct('FeatureReductor',FeatureReductor,'Classifier',Classifier,'Accuracy',Accuracy);

% Adhoc Plot for KSVM  classify(without feature reduction)
DimM=1; 
FeatureReductor='NONE';
Accuracy=AdhocAnalysis(ProjectedData,Y_Train,Y_Test_Desired,FeatureReductor,1,Classifier);
AcaPlot(17)=struct('FeatureReductor',FeatureReductor,'Classifier',Classifier,'Accuracy',Accuracy);

% %% 
% %%------------------SVM  classify
% 
% Classifier='SVM'; %kernel SVM (Kernel function rbf)
% 
% % Adhoc Plot for LDA--SVM  classify
% DimM=(NoC-1);  %exam dimension from 1 to DimM
% FeatureReductor='LDA';
% Accuracy=AdhocAnalysis(ProjectedData,Y_Train,Y_Test_Desired,FeatureReductor,DimM,Classifier);
% AcaPlot(18)=struct('FeatureReductor',FeatureReductor,'Classifier',Classifier,'Accuracy',Accuracy);
%  
% % Adhoc Plot for PCA--SVM  classify
% DimM=50;  %exam dimension from 1 to DimM
% FeatureReductor='PCA';
% Accuracy=AdhocAnalysis(ProjectedData,Y_Train,Y_Test_Desired,FeatureReductor,DimM,Classifier);
% AcaPlot(19)=struct('FeatureReductor',FeatureReductor,'Classifier',Classifier,'Accuracy',Accuracy);
% 
% % Adhoc Plot for KLDA--SVM  classify
% DimM=(NoC-1);  %exam dimension from 1 to DimM
% FeatureReductor='KLDA';
% Accuracy=AdhocAnalysis(ProjectedData,Y_Train,Y_Test_Desired,FeatureReductor,DimM,Classifier);
% AcaPlot(20)=struct('FeatureReductor',FeatureReductor,'Classifier',Classifier,'Accuracy',Accuracy);
% 
% % Adhoc Plot for KPCA--SVM  classify
% DimM=50;  %exam dimension from 1 to DimM
% FeatureReductor='KPCA';
% Accuracy=AdhocAnalysis(ProjectedData,Y_Train,Y_Test_Desired,FeatureReductor,DimM,Classifier);
% AcaPlot(21)=struct('FeatureReductor',FeatureReductor,'Classifier',Classifier,'Accuracy',Accuracy);
% 
% % Adhoc Plot for SVM  classify(without feature reduction)
% DimM=1; 
% FeatureReductor='NONE';
% Accuracy=AdhocAnalysis(ProjectedData,Y_Train,Y_Test_Desired,FeatureReductor,1,Classifier);
% AcaPlot(22)=struct('FeatureReductor',FeatureReductor,'Classifier',Classifier,'Accuracy',Accuracy);

%%
 save -v7.3 '.\Results\AcaPlot.mat' AcaPlot;
% %storage the feature reducted data set into Results folder
























%%
% 
% % --------------------------
% %find best dimension for PCA--KNN  classify
% Accuracy=zeros(1,NoF);
% for Dim=1:NoF
%    
%     %Feature reduction: PCA
%     [X_Train_PCAProj,X_Test_PCAProj] = PCAProj(X_Train,X_Test, Dim);
%      %Classifier: KNNClassifier 
%      [Y_Test] = KNNClassifier(X_Train_LDAProj, Y_Train, X_Test_LDAProj);
%     
%     Accuracy(Dim)=  sum(Y_Test_Desired==Y_Test)/ length(X_Test);
% end 
% 
% figure(2)
% [MaxAccuracy,Ind_MaxAccuracy] = max(Accuracy);
% xaxis= 1:NoF;
% plot(xaxis,Accuracy);grid on
% xlabel('Number of Projected Dimension'),ylabel('Accuracy')
% title({['The performance of  PCA-- KNN,'];['best on Dim=',num2str(Ind_MaxAccuracy),'  Highest Accuarcy=',num2str(MaxAccuracy*100),'%']})
% %savefig(' PCAProj-KNN.fig')   
% %--------------------------

% %find best dimension for LDA--KNN  classify  
% Accuracy=zeros(1,(NoC-1));
% for Dim=1:(NoC-1)
%         %Feature reduction: LDA
%     [X_Train_LDAProj,X_Test_LDAProj] = FLDAProj(X_Train, Y_Train, X_Test, Dim);
%     %Classifier: KNNClassifier 
%     [Y_Test] = KNNClassifier(X_Train_LDAProj, Y_Train, X_Test_LDAProj);
%     
%     Accuracy(Dim)=  sum(Y_Test_Desired==Y_Test)/ length(X_Test);
% end 
% 
% figure(3)
% [MaxAccuracy,Ind_MaxAccuracy] = max(Accuracy);
% xaxis= 1:(NoC-1);
% plot(xaxis,Accuracy);grid on
% xlabel('Number of Projected Dimension'),ylabel('Accuracy')
% title({['The performance of  LDA-- KNNclassifer,'];['best on Dim=',num2str(Ind_MaxAccuracy),'  Highest Accuarcy=',num2str(MaxAccuracy*100),'%']})
% %savefig(' LDA-- KNNclassifer.fig')   


% % %--------------------------
% % find best dimension for KLDA--KNN  classify  
% % best on Dim=(NoC-1), acuracy=65%
% Accuracy=zeros(1,(NoC-1));
%  for Dim=1:(NoC-1)
%     %    Feature reduction: KLDA
% %     addpath('C:\Users\xli63\Downloads\KEDcode\KEDcode_yest')
%    [X_Train_LDAProj,X_Test_LDAProj] = KLDA_proj(X_Train', Y_Train, X_Test', Dim);
% 
%     
% %     Classifier: KNNClassifier 
%     [Y_Test] = KNNClassifier(X_Train_LDAProj, Y_Train, X_Test_LDAProj);
%     
%     Accuracy(Dim)=  sum(Y_Test_Desired==Y_Test)/ length(X_Test);
%  end 
% 
% figure(4)
% [MaxAccuracy,Ind_MaxAccuracy] = max(Accuracy);
% xaxis= 1:Dim;
% plot(xaxis,Accuracy(1:Dim));grid on
% xlabel('Number of Projected Dimension'),ylabel('Accuracy')
% title({['The performance of  kernal LDA-- KNNclassifer,'];['best on Dim=',num2str(Ind_MaxAccuracy),'  Highest Accuarcy=',num2str(MaxAccuracy*100),'%']})
% savefig(' KLDA-- KNNclassifer.fig')   


% %--------------------------
%find best dimension for KLDA--GaussianMLClassifier 
%best on Dim=9 ACCU =45%
% Accuracy=zeros(1,(NoC-1));
% for Dim=1:(NoC-1)
%    
%         %Feature reduction: KLDA
%     
%    [X_Train_LDAProj,X_Test_LDAProj] = KLDA_proj(X_Train', Y_Train, X_Test', Dim);
% 
%      %Classifier: GaussianMLClassifier 
%   [Y_Test] = GaussianMLClassifier3(X_Train_LDAProj', Y_Train, X_Test_LDAProj');
%     
%     Accuracy(Dim)=  sum(Y_Test_Desired==Y_Test)/ length(X_Test);
% end 
% 
% figure(5)
% [MaxAccuracy,Ind_MaxAccuracy] = max(Accuracy);
% xaxis= 1:Dim;
% plot(xaxis,Accuracy);grid on
% xlabel('Number of Projected Dimension'),ylabel('Accuracy')
% title({['The performance of  kernel LDA-- GaussianMLClassifier,'];['best on Dim=',num2str(Ind_MaxAccuracy),'  Highest Accuarcy=',num2str(MaxAccuracy*100),'%']})
% %savefig(' KLDA-- GaussianMLClassifier.fig')   

% %--------------------------
% find best dimension for KPCA--KNN  classify  
%best on 
% Accuracy=zeros(1,(NoC-1));
%  for Dim=1:NoF
%      Feature reduction: KPCA
%     addpath('C:\Users\xli63\Downloads\KEDcode\KEDcode_yest')
%    [X_Train_LDAProj,X_Test_LDAProj] = KPCA_proj(X_Train,  X_Test, Dim);
% 
%     
%     Classifier: KNNClassifier 
%     [Y_Test] = KNNClassifier(X_Train_LDAProj, Y_Train, X_Test_LDAProj);
%     
%     Accuracy(Dim)=  sum(Y_Test_Desired==Y_Test)/ length(X_Test);
% end 
% 
% figure(6)
% [MaxAccuracy,Ind_MaxAccuracy] = max(Accuracy);
% xaxis= 1:Dim;
% plot(xaxis,Accuracy(1:Dim));grid on
% xlabel('Number of Projected Dimension'),ylabel('Accuracy')
% title({['The performance of  kernal PCA-- KNNclassifer,'];['best on Dim=',num2str(Ind_MaxAccuracy),'  Highest Accuarcy=',num2str(MaxAccuracy*100),'%']})
% %savefig(' KPCA-- KNNclassifer.fig')   

% %--------------------------
% find best dimension for KPCA--GaussianMLClassifier
%best on Dim=18 51%
% Accuracy=zeros(1,(NoC-1));
%  for Dim=1:NoF
%   %    Feature reduction: KPCA
% %     addpath('C:\Users\xli63\Downloads\KEDcode\KEDcode_yest')
%    [X_Train_LDAProj,X_Test_LDAProj] = KPCA_proj(X_Train,  X_Test, Dim);
% 
%     
% %     Classifier: GaussianMLClassifier 
%     [Y_Test] = GaussianMLClassifier3(X_Train_LDAProj', Y_Train, X_Test_LDAProj');
%     
%     Accuracy(Dim)=  sum(Y_Test_Desired==Y_Test)/ length(X_Test);
% end 
% % 
% figure(7)
% [MaxAccuracy,Ind_MaxAccuracy] = max(Accuracy);
% xaxis= 1:Dim;
% plot(xaxis,Accuracy(1:Dim));grid on
% xlabel('Number of Projected Dimension'),ylabel('Accuracy')
% title({['The performance of  kernal PCA-- GaussianMLClassifier,'];['best on Dim=',num2str(Ind_MaxAccuracy),'  Highest Accuarcy=',num2str(MaxAccuracy*100),'%']})
% %savefig(' KPCA-- GaussianMLClassifier.fig')   


% % % %--------------------------
% % find best dimension for KLDA--SVM  classify  
% % best on Dim=(NoC-1), acuracy=65%
% Accuracy=zeros(1,(NoC-1));
%  for Dim=1:(NoC-1)
% %    Feature reduction: KLDA
% %     addpath('C:\Users\xli63\Downloads\KEDcode\KEDcode_yest')
%    [X_Train_KLDAProj,X_Test_KLDAProj] = KLDA_proj(X_Train', Y_Train, X_Test', Dim);
% 
%     %     Classifier: SVM
%      t = templateSVM('Standardize',1,'KernelFunction','rbf');
%     Mdl = fitcecoc(X_Train_KLDAProj,Y_Train,'Learners',t,'Coding','onevsall');
%     Y_Test=predict(Mdl,X_Test_KLDAProj);
%     Accuracy(Dim)=  sum(Y_Test_Desired==Y_Test)/ length(X_Test);
%  end 
% 
% figure(8)
% [MaxAccuracy,Ind_MaxAccuracy] = max(Accuracy);
% xaxis= 1:Dim;
% plot(xaxis,Accuracy(1:Dim));grid on
% xlabel('Number of Projected Dimension'),ylabel('Accuracy')
% title({['The performance of  kernelLDA-- SVMclassifer,'];['best on Dim=',num2str(Ind_MaxAccuracy),'  Highest Accuarcy=',num2str(MaxAccuracy*100),'%']})
%savefig(' KLDA-- SVMclassifer.fig')   

% %--------------------------
%find best dimension for LDA--SVM  classify  
% % best on Dim=(NoC-1), acuracy=65%
% Accuracy=zeros(1,(NoC-1));
%  for Dim=1:(NoC-1)     
% %    Feature reduction: LDA
%     [X_Train_LDAProj,X_Test_LDAProj] = FLDAProj(X_Train, Y_Train, X_Test, Dim);
%     %     Classifier: SVM
%      t = templateSVM('Standardize',1,'KernelFunction','rbf');
%     Mdl = fitcecoc(X_Train_LDAProj,Y_Train,'Learners',t,'Coding','onevsall');
%     Y_Test=predict(Mdl,X_Test_LDAProj);
%     
%     Accuracy(Dim)=  sum(Y_Test_Desired==Y_Test)/ length(X_Test);
%  end 
% 
% figure(9)
% [MaxAccuracy,Ind_MaxAccuracy] = max(Accuracy);
% xaxis= 1:Dim;
% plot(xaxis,Accuracy);grid on
% xlabel('Number of Projected Dimension'),ylabel('Accuracy')
% title({['The performance of  LDA-- SVMclassifer,'];['best on Dim=',num2str(Ind_MaxAccuracy),'  Highest Accuarcy=',num2str(MaxAccuracy*100),'%']})
% savefig(' LDA-- SVMclassifer.fig')   

% % --------------------------
% %find best dimension for PCA--SVM  classify
% Accuracy=zeros(1,NoF);
% for Dim=1:NoF
%    
%     %Feature reduction: PCA
%     [X_Train_PCAProj,X_Test_PCAProj] = PCAProj(X_Train,X_Test, Dim);
%      %     Classifier: SVM
%      t = templateSVM('Standardize',1,'KernelFunction','rbf');
%     Mdl = fitcecoc(X_Train_PCAProj,Y_Train,'Learners',t,'Coding','onevsall');
%     Y_Test=predict(Mdl,X_Test_PCAProj);
%     
%     
%     Accuracy(Dim)=  sum(Y_Test_Desired==Y_Test)/ length(X_Test);
% end 
% 
% figure(10)
% [MaxAccuracy,Ind_MaxAccuracy] = max(Accuracy);
% xaxis= 1:NoF;
% plot(xaxis,Accuracy);grid on
% xlabel('Number of Projected Dimension'),ylabel('Accuracy')
% title({['The performance of  PCA-- SVM,'];['best on Dim=',num2str(Ind_MaxAccuracy),'  Highest Accuarcy=',num2str(MaxAccuracy*100),'%']})
% savefig(' PCA-SVM.fig')   
%--------------------------