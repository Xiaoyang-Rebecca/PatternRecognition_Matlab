function [X_Train_Proj,X_Test_Proj] = GDA_proj(X_Train, Y_Train, X_Test, Dim)
addpath('.\drtoolbox\techniques')

    [X_Train_Proj,eigenvector] = gda(X_Train, Y_Train, Dim, 'poly',1,2);
    
    % Compute kernel matrix o testing data
    disp('Computing kernel matrix...');
    K = gram(X_Test, X_Train,'poly',1,2);

    % Perform eigenvector decomposition of kernel matrix (Kc = P * gamma * P')
    disp('Performing eigendecomposition of kernel matrix...');
    K(isnan(K)) = 0;
    K(isinf(K)) = 0;
    
    [P, gamma,~] = svd(K);

	if size(P, 2) < size(X_Test,1)
		error('Singularities in kernel matrix prevent solution.');
    end
	
	% Sort eigenvalues and vectors in descending order
	[gamma, ind] = sort(diag(gamma), 'descend');
	P = P(:,ind);

	% Remove eigenvectors with relatively small value
	minEigv = max(gamma) / 1e5;
	ind = find(gamma > minEigv);
	P = P(:,ind);
    
	% Remove eigenvectors with relatively small value
	minEigv = max(gamma) / 1e5;
	ind = find(gamma > minEigv);
	P = P(:,ind);
	gamma = gamma(ind);
	rankK = length(ind);
    
	% Recompute kernel matrix
    K = P * diag(gamma) * P';

    
    
    
    X_Test_Proj = P * eigenvector; 
    
    
    