     function [X_Train_Proj,X_Test_Proj]=FeatureReduction(X_Train,Y_Train,X_Test,FeatureReductor,Dim)
        if strcmp(FeatureReductor,'LDA')   ==1
            [X_Train_Proj,X_Test_Proj] = FLDAProj(X_Train, Y_Train, X_Test, Dim);
        elseif strcmp(FeatureReductor,'PCA')   ==1 
            [X_Train_Proj,X_Test_Proj] = PCAProj(X_Train,X_Test, Dim);
        elseif  strcmp(FeatureReductor,'KLDA')  ==1  
            [X_Train_Proj,X_Test_Proj] = KLDA_proj(X_Train', Y_Train, X_Test',Dim);
        elseif  strcmp(FeatureReductor,'KPCA')  ==1  
            [X_Train_Proj,X_Test_Proj] = KPCA_proj(X_Train,  X_Test, Dim);
         elseif  strcmp(FeatureReductor,'NONE')  ==1 
            X_Train_Proj=X_Train';
            X_Test_Proj = X_Test';
        end
    end