function [X_Train_PCAProj,X_Test_PCAProj] = KPCA_proj(X_Train, X_Test, Dim)
% addpath('.\KEDcode')
% addpath('.\drtoolbox\techniques')
% % 
% options.KernelType = 'Gaussian';
% %Sigma = 1;
% 
% 
data=[X_Train X_Test]';  %NoSample *NoFeature
% 
% 
% %KPCA: Project Testing data and training data together
% options.t = 1;
% options.ReducedDim=Dim;


% 

addpath('.\KEDcode')

options.KernelType = 'Polynomial';
options.d= 2;

[eigvector, ~,K] = KPCA(data, options,Dim);
% 
% 

X_PCA_Proj= K * eigvector;

%[mappedX, mapping] = kernel_pca(X, no_dims, kernel, param1, param2)
%[X_PCA_Proj, ~] = kernel_pca(data, Dim,'poly',1,2);

X_Train_PCAProj=X_PCA_Proj(1:size(X_Train,2),:);

X_PCA_Proj(1:size(X_Train,2),:)=[];
X_Test_PCAProj=X_PCA_Proj;



end


%%
%------------------example
%           options.KernelType = 'Gaussian';
%           options.t = 1;
%           options.ReducedDim = 4;
% 			fea = rand(7,10);   %7sample,10 dimension
% 			[eigvector,eigvalue] = KPCA(fea,options);
% 
%           feaTest = rand(3,10);
%           Ktest = constructKernel(feaTest,fea,options)
%           Y = Ktest*eigvector;