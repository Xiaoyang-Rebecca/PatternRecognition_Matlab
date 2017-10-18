function [X_Train_LDAProj,X_Test_LDAProj] = KLDA_proj(X_Train, Y_Train, X_Test, Dim)
addpath('.\KEDcode')
options.KernelType = 'Polynomial';
options.d= 2;

[eigvector, eigvalue,K_train] = KDA(options,Y_Train,X_Train,Dim);

% [~,ind_SortedEigVal] = sort(eigvalue,'descend');  %get the largest 1~dim of eigvalues 
%  New_U = eigvector(:,ind_SortedEigVal(1:Dim));    %get part of eigen vector
%projct the gram  
X_Train_LDAProj=K_train*eigvector;

[K_test,~] = constructKernel(X_Test,X_Train,options);
X_Test_LDAProj=K_test*eigvector;

