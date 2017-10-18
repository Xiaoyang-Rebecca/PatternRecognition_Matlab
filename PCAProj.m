function [X_Train_Proj,X_Test_Proj] = PCAProj(X_Train,X_Test, Dim)
	%Dim: dimensionality of the projected subspace 
    %Dim<= NoF

	[~  ,NoTrS] = size(X_Train); 
	[NoF,NoTeS] = size(X_Test);

	%%zero-mean the Training and Test data 
	e_XTrain = mean (X_Train,2);
	e_XTest  = mean (X_Test ,2);

	for i = 1: NoTrS
		X_Train_centered(:,i) = X_Train(:,i) - e_XTrain;
	end

	for i = 1: NoTeS
		X_Test_centered(:,i)  = X_Test(:,i)  - e_XTrain;
    end
    
%     X_Train_Proj = PCA(X_Train_centered,Dim,NoTrS);
%     X_Test_Proj  = PCA(X_Test_centered ,Dim,NoTrS);


    
    X=X_Train_centered;
    %Define a scatter matrix
	S = 0;
	for k = 1: NoTrS
		S = S + X(:,k)*(X(:,k))';
	end
    
	%Decomposition of a scatter matrix  S = EigVec * EigVal * EigVec'
	[EigVec,EigVal] = eig(S)  ;
    [~,ind_SortedEigVal] = sort(diag(EigVal),'descend');
	
	New_U = EigVec(:,ind_SortedEigVal(1:Dim));%select d Eigen Vectors corresponding to d highest Eigen values

    
	X_Train_Proj = (New_U'* X_Train_centered )';
    X_Test_Proj  = (New_U'* X_Test_centered )';
    
    
    
 end

