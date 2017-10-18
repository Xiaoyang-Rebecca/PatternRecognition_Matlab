

load('data.mat')


addpath('.//KEDcode')
addpath('.//SVM')
addpath('.//GMM')

X_Train = data.Xtrain';  %750*NoF 750 samples,NoF features
Y_Train = data.Ytrain;  %750*1   750 samples,target=1..15 (15 classes)

X_Test  = data.Xval';      %12197*NoF
Y_Test_Desired  = data.Yval;   %the desire output of target(class value)

%down sample for testing data: select a designated sample per
%class-------------------



Dim=14;K=2;
% 
% %FeatureReductor='LDA';  %Dim<15
% %FeatureReductor='PCA';
% %FeatureReductor='KLDA'; %Dim<15
% %FeatureReductor='KPCA';
% %FeatureReductor='NONE';
% 
% %Classifier='GaussianML';
% %Classifier='KNN';
% %Classifier='GMM';
% %Classifier='KSVM';

FeatureReductor='PCA';Classifier='GaussianML';      
Accuracy(1,1) = PatternRecog(X_Train,Y_Train,X_Test,Y_Test_Desired,FeatureReductor,Dim,Classifier,K);
FeatureReductor='PCA';Classifier='GMM';      
Accuracy(2,1) = PatternRecog(X_Train,Y_Train,X_Test,Y_Test_Desired,FeatureReductor,Dim,Classifier,K);
FeatureReductor='PCA';Classifier='KNN';      
Accuracy(3,1) = PatternRecog(X_Train,Y_Train,X_Test,Y_Test_Desired,FeatureReductor,Dim,Classifier,K);
FeatureReductor='PCA';Classifier='KSVM';      
Accuracy(4,1) = PatternRecog(X_Train,Y_Train,X_Test,Y_Test_Desired,FeatureReductor,Dim,Classifier,K);

FeatureReductor='KPCA';Classifier='GaussianML';      
Accuracy(1,2) = PatternRecog(X_Train,Y_Train,X_Test,Y_Test_Desired,FeatureReductor,Dim,Classifier,K);
FeatureReductor='KPCA';Classifier='GMM';      
Accuracy(2,2) = PatternRecog(X_Train,Y_Train,X_Test,Y_Test_Desired,FeatureReductor,Dim,Classifier,K);
FeatureReductor='KPCA';Classifier='KNN';      
Accuracy(3,2) = PatternRecog(X_Train,Y_Train,X_Test,Y_Test_Desired,FeatureReductor,Dim,Classifier,K);
FeatureReductor='KPCA';Classifier='KSVM';      
Accuracy(4,2) = PatternRecog(X_Train,Y_Train,X_Test,Y_Test_Desired,FeatureReductor,Dim,Classifier,K);

FeatureReductor='LDA';Classifier='GaussianML';      
Accuracy(1,3) = PatternRecog(X_Train,Y_Train,X_Test,Y_Test_Desired,FeatureReductor,Dim,Classifier,K);
FeatureReductor='LDA';Classifier='GMM';      
Accuracy(2,3) = PatternRecog(X_Train,Y_Train,X_Test,Y_Test_Desired,FeatureReductor,Dim,Classifier,K);
FeatureReductor='LDA';Classifier='KNN';      
Accuracy(3,3) = PatternRecog(X_Train,Y_Train,X_Test,Y_Test_Desired,FeatureReductor,Dim,Classifier,K);
FeatureReductor='LDA';Classifier='KSVM';      
Accuracy(4,3) = PatternRecog(X_Train,Y_Train,X_Test,Y_Test_Desired,FeatureReductor,Dim,Classifier,K);


FeatureReductor='KLDA';Classifier='GaussianML';      
Accuracy(1,4) = PatternRecog(X_Train,Y_Train,X_Test,Y_Test_Desired,FeatureReductor,Dim,Classifier,K);
FeatureReductor='KLDA';Classifier='GMM';      
Accuracy(2,4) = PatternRecog(X_Train,Y_Train,X_Test,Y_Test_Desired,FeatureReductor,Dim,Classifier,K);
FeatureReductor='KLDA';Classifier='KNN';      
Accuracy(3,4) = PatternRecog(X_Train,Y_Train,X_Test,Y_Test_Desired,FeatureReductor,Dim,Classifier,K);
FeatureReductor='KLDA';Classifier='KSVM';      
Accuracy(4,4) = PatternRecog(X_Train,Y_Train,X_Test,Y_Test_Desired,FeatureReductor,Dim,Classifier,K);

Accuracy_downsample_Matrix(:,:,1)=Accuracy;
Accuracy_downsample_Vector(1,:)=Accuracy(:);
% change the size of training dataset:----------------------------------------


i=1;
for NoS=[10,20,30]
    i=i+1;
    [DS_X_Test,DS_Y_Test_Desired]= downsample(X_Test,Y_Test_Desired,NoS);
	FeatureReductor='PCA';Classifier='GaussianML';      
	Accuracy(1,1) = PatternRecog(X_Train,Y_Train,DS_X_Test,DS_Y_Test_Desired,FeatureReductor,Dim,Classifier,K);
	FeatureReductor='PCA';Classifier='GMM';      
	Accuracy(2,1) = PatternRecog(X_Train,Y_Train,DS_X_Test,DS_Y_Test_Desired,FeatureReductor,Dim,Classifier,K);
	FeatureReductor='PCA';Classifier='KNN';      
	Accuracy(3,1) = PatternRecog(X_Train,Y_Train,DS_X_Test,DS_Y_Test_Desired,FeatureReductor,Dim,Classifier,K);
	FeatureReductor='PCA';Classifier='KSVM';      
	Accuracy(4,1) = PatternRecog(X_Train,Y_Train,DS_X_Test,DS_Y_Test_Desired,FeatureReductor,Dim,Classifier,K);

	FeatureReductor='KPCA';Classifier='GaussianML';      
	Accuracy(1,2) = PatternRecog(X_Train,Y_Train,DS_X_Test,DS_Y_Test_Desired,FeatureReductor,Dim,Classifier,K);
	FeatureReductor='KPCA';Classifier='GMM';      
	Accuracy(2,2) = PatternRecog(X_Train,Y_Train,DS_X_Test,DS_Y_Test_Desired,FeatureReductor,Dim,Classifier,K);
	FeatureReductor='KPCA';Classifier='KNN';      
	Accuracy(3,2) = PatternRecog(X_Train,Y_Train,DS_X_Test,DS_Y_Test_Desired,FeatureReductor,Dim,Classifier,K);
	FeatureReductor='KPCA';Classifier='KSVM';      
	Accuracy(4,2) = PatternRecog(X_Train,Y_Train,DS_X_Test,DS_Y_Test_Desired,FeatureReductor,Dim,Classifier,K);

	FeatureReductor='LDA';Classifier='GaussianML';      
	Accuracy(1,3) = PatternRecog(X_Train,Y_Train,DS_X_Test,DS_Y_Test_Desired,FeatureReductor,Dim,Classifier,K);
	FeatureReductor='LDA';Classifier='GMM';      
	Accuracy(2,3) = PatternRecog(X_Train,Y_Train,DS_X_Test,DS_Y_Test_Desired,FeatureReductor,Dim,Classifier,K);
	FeatureReductor='LDA';Classifier='KNN';      
	Accuracy(3,3) = PatternRecog(X_Train,Y_Train,DS_X_Test,DS_Y_Test_Desired,FeatureReductor,Dim,Classifier,K);
	FeatureReductor='LDA';Classifier='KSVM';      
	Accuracy(4,3) = PatternRecog(X_Train,Y_Train,DS_X_Test,DS_Y_Test_Desired,FeatureReductor,Dim,Classifier,K);


	FeatureReductor='KLDA';Classifier='GaussianML';      
	Accuracy(1,4) = PatternRecog(X_Train,Y_Train,DS_X_Test,DS_Y_Test_Desired,FeatureReductor,Dim,Classifier,K);
	FeatureReductor='KLDA';Classifier='GMM';      
	Accuracy(2,4) = PatternRecog(X_Train,Y_Train,DS_X_Test,DS_Y_Test_Desired,FeatureReductor,Dim,Classifier,K);
	FeatureReductor='KLDA';Classifier='KNN';      
	Accuracy(3,4) = PatternRecog(X_Train,Y_Train,DS_X_Test,DS_Y_Test_Desired,FeatureReductor,Dim,Classifier,K);
	FeatureReductor='KLDA';Classifier='KSVM';      
	Accuracy(4,4) = PatternRecog(X_Train,Y_Train,DS_X_Test,DS_Y_Test_Desired,FeatureReductor,Dim,Classifier,K);
    
    Accuracy_downsample_Matrix(:,:,i)=Accuracy;
    Accuracy_downsample_Vector(i,:)=Accuracy(:);
end


%%
%---------------choose a Feature reduction method:

%FeatureReductor='LDA';  %Dim<15
%FeatureReductor='PCA';
%FeatureReductor='KLDA'; %Dim<15
%FeatureReductor='KPCA';
%FeatureReductor='NONE';





%Parameter:Dim
%Dim=14;

  
%[X_Train_Proj,X_Test_Proj]=FeatureReduction(X_Train,Y_Train,X_Test,FeatureReductor,Dim);
    
%%
%feature Classification:----------------choose a Classifier:

    targets=ind2vec(Y_Test_Desired');
    outputs=ind2vec(Y_Test');
    plotconfusion(targets,outputs);
    plotroc(targets,outputs);
 
