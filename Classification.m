    function Y_Test=Classification(X_Train_Proj, Y_Train, X_Test_Proj,Classifier,K)
        if strcmp(Classifier,'GaussianML' )==1
            [Y_Test] = GaussianMLClassifier3(X_Train_Proj', Y_Train, X_Test_Proj');
        elseif strcmp(Classifier,'KNN' )==1
            [Y_Test] = KNNClassifier(X_Train_Proj, Y_Train, X_Test_Proj,K);
        elseif strcmp(Classifier,'SVM' )==1
            t = templateSVM('Standardize',1,'linear');
            Mdl = fitcecoc(X_Train_Proj,Y_Train,...
                'Learners',t,'Coding','onevsall');
            Y_Test=predict(Mdl,X_Test_Proj);   
        elseif strcmp(Classifier,'KSVM' )==1
            t = templateSVM('Standardize',1,'KernelFunction','rbf');
            Mdl = fitcecoc(X_Train_Proj,Y_Train,...
                'Learners',t,'Coding','onevsall');
            Y_Test=predict(Mdl,X_Test_Proj);   
        elseif strcmp(Classifier,'GMM' )==1
            Y_Test = GMMClassifier( X_Train_Proj, Y_Train,X_Test_Proj,K);    
        end 
    end