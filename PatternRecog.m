function Accuracy=PatternRecog(X_Train,Y_Train,X_Test,Y_Test_Desired,FeatureReductor,Dim,Classifier,K)
    
    [X_Train_Proj,X_Test_Proj]=FeatureReduction(X_Train,Y_Train,X_Test,FeatureReductor,Dim);
    Y_Test=Classification(X_Train_Proj, Y_Train, X_Test_Proj,Classifier,K);
    Accuracy=  sum(Y_Test_Desired==Y_Test)/ length(X_Test);
    
end
