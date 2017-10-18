function AoC = ML_Classify(NoS)
%input: Number of Sample 
%output: accuracy of classifying



%-------------[Step 2]-------------
%--------------------Get the training dataset and testing dataset--------------
%Partition your available labeled data in data.mat into two halves for each class – the first half
%(comprising of the first 500 samples for each class) should be used for “training” and the
%other half (comprising of the next 500 samples for each class) 
load data.mat
Samp_Index = randperm(500,NoS);  % generate NoS unique random numbers(from 1:500) correspond to the index of samples

f1_train = data.f1(Samp_Index,:);
f2_train = data.f2(Samp_Index,:);
f3_train = data.f3(Samp_Index,:);

f1_test = data.f1(500+Samp_Index,:);
f2_test = data.f2(500+Samp_Index,:);
f3_test = data.f3(500+Samp_Index,:);

X_Train = [f1_train ;f2_train; f3_train]';  %size of training data= 3*(500+500+500)
Z=zeros(NoS,1);  
Y_Train = [(Z+1)    ;(Z+2)   ;(Z+3)   ]; %  labels of training data:lable "i" for class i  [(500+500+500) *1]
X_Test  = [f1_test  ;f2_test ; f3_test]'; 

%--------------------Get the (result)label of testing dataset-------------
%input training data, labels of training data, testing data ; output labels of testing data

[Y_Test] = GaussianMLClassifier3(X_Train , Y_Train, X_Test);  

%-------------[Step 3]-------------
%--------------------estimate the accuracy of your classifier  ----------
%ratio: Number of test samples correctly classified / total number of test samples.
Idx_cor_classified = find(Y_Train==Y_Test); %test samples correctly classified 
AoC =  length(Idx_cor_classified)./ length(X_Test);


