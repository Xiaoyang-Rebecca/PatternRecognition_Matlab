%% Simple test of matlab wrappers for MSVMpack
%
%	3 categories with 100 data points each. 
%

% Make data set
disp('>>> Generating a data set (X,Y) for 3 categories with 100 data points each.');

X = zeros(300, 2);
X(1:100,:) = randn(100,2) + ones(100,1)* [1,1];
X(101:200,:) = randn(100,2) + ones(100,1)*[-1,-1];
X(201:300,:) = randn(100,2) + ones(100,1)*[1,-1];

Y = ones(300,1); 
Y(101:200) = 2 * ones(100,1); 
Y(201:300) = 3 * ones(100,1); 

% Train model
disp('  ');
disp('>>> Train an MSVM (Weston & Watkins type) with linear kernel and default parameters:');
disp('       [model] = trainmsvm(X,Y,''-m WW -k 1'', ''mymsvm'') ');
disp('  ');
[model] = trainmsvm(X,Y,'-m WW -k 1', 'mymsvm');

if isstruct(model)

	% Make predictions with the model in memory
	disp('  ');
	disp('>>> Predict the labels with the MSVM model:');
	disp('       [labels, outputs] = predmsvm(model, X,Y);');
	disp('  ');
	[labels, outputs] = predmsvm(model, X,Y);

	% or with a model in a file with its modelname:
	disp('  ');
	disp('>>> alternatively, predict the labels with the MSVM model in a file via its name:');
	disp('       [labels, outputs] = predmsvm(''mymsvm'', X,Y)');
	disp('  ');
	[labels, outputs] = predmsvm('mymsvm', X,Y);
	
	% Load a model from a file and display information
	disp('  ');
	disp('>>> Load a model from a file and display information:');
	disp('        model = loadmsvm(''mymsvm'')');
	model = loadmsvm('mymsvm');

	% Plot the results
	figure;
	hold on;
	plot(X(Y==1,1), X(Y==1,2), 'ob'); 
	plot(X(Y==2,1), X(Y==2,2), '+r'); 
	plot(X(Y==3,1), X(Y==3,2), '*g'); 
	title('Training data set'); 

	figure;
	hold on;
	plot(X(labels==1,1), X(labels==1,2), 'ob'); 
	plot(X(labels==2,1), X(labels==2,2), '+r'); 
	plot(X(labels==3,1), X(labels==3,2), '*g'); 
	title('Labels predicted by the MSVM model on the training set'); 

	% Cross validation
	disp('  ');
	disp('>>> Perform a 5-fold cross validation:');
	disp('         [cv_error, cv_labels] = kfold(5, X, Y, ''-m WW -k 1'')');
	disp('  ');	disp('  ');
	[cv_error, cv_labels] = kfold(5, X, Y, '-m WW -k 1');
	disp(sprintf('Cross-validation error = %f (should be around 0.2)\n',cv_error));
	disp('''cv_labels'' can be used to compute other statistics than the mean error.');
	
else
	% Error during call to trainmsvm
	disp('** You must build MSVMpack before using the wrappers for matlab **');
	disp('** I will try to do it for you... **');
	
	command = sprintf('cd %s ; make', msvmpackdir);
	[status,cmdout] = system(command,'-echo');
	
	disp('*******************************************');
	disp('** You can retry to run "example.m" now. **');
	disp('*******************************************');
end
