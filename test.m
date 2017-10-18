load 'D:\materials of courses of Rebecca\Digital Pattern Recognition\HWS\data.mat'
f1_train = data.f1(1:500,:);
f2_train = data.f2(1:500,:);
f3_train = data.f3(1:500,:);

f1_test = data.f1(501:1000,:);
f2_test = data.f2(501:1000,:);
f3_test = data.f3(501:1000,:);

X_Train = [f1_train ;f2_train; f3_train]';  %size of training data= 3*(500+500+500)
Z=zeros(500,1);  
Y_Train = [(Z+1)    ;(Z+2)   ;(Z+3)   ]; %  labels of training data:lable "i" for class i  [(500+500+500) *1]

X_Test  = [f1_test  ;f2_test ; f3_test]'; %3*1500
Y_Test_True = Y_Train;   % the true state of art of the labels of testing data


%[idx,C] = kmeans(X_Test',4) %returns the k cluster centroid locations in the k-by-p matrix C.


   [X_Train_PCAProj,X_Test_PCAProj] = KPCA_proj(X_Train, X_Test, 2);
    %[Y_Test] = GMMClassifier( X_Train_PCAProj', Y_Train,X_Test_PCAProj', 1 );  


%C : k cluster centroid locations in the k-by-p matrix C
%centroid of cluster i is C(i,:)


% Dim=2
% %[X_Train_LDAProj,X_Test_LDAProj] = BettyKLDA_proj(X_Train', Y_Train, X_Test', Dim);
% 
%[Y_Test] = KNNClassifier(X_Train', Y_Train, X_Test',2);
[Y_Test] = KNNClassifier(X_Train_PCAProj, Y_Train, X_Test_PCAProj,2);
% 
Accuracy=  sum(Y_Test_True==Y_Test)/ length(X_Test);
 warning('off','all')
%  
%  

 