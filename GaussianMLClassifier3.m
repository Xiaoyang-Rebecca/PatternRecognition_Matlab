%Copyright: Xiaoyang LI
%Feb,2016
%[Y_Test] = GaussianMLClassifier3[X_Train, Y_Train, X_Test]
%-----------------[Step 1]
%Input:  
%     X_Train is a data matrix [dimensions number of features x 
%number of training samples] of all classes in the training data (the data 
%that we will use to “learn” the classifier.)  [NoF * NoTrS]  X:data
%     Y_Train is a label vector of   [size number of training samples x 1], 
%where the i’th entry in Y_Train contains the label information (an integer
%value indicating the class index) for the i’th data sample from the data 
%matrix X_Train. [NoTrS * 1] Y:label(Class wi) [NoTrS *1]
%     X_Test is a data matrix of [size number of features x number of testing 
%samples]. [NoF * NoTeS] X:data

%Output: 
%     Y_Test is the “predicted” label (or labels if X_Test has multiple test
%samples) that your classifier determines based on the log likelihood 
%discriminant functions for the test sample(s). 
%

function [Y_Test] = GaussianMLClassifier3(X_Train, Y_Train, X_Test)
%Training procedure: using the training data to determine the parameters
%using the ML estimator for the Gaussian case.

[~  ,NoTrS] = size(X_Train); 
[NoF,NoTeS] = size(X_Test);
Y_Test=zeros(NoTeS,1);

%%---------------------Get the class labels---------------------------
%Assume that the data is labelled in continuous integer: 1,2,3...,Noc
%NoC = max(Y_Train);   %Number of class(label)

%If the data is not labelled in continuous integers,but 0,2,5,7,8...
% we use class(NoC:,2) to display the attribute of class
class_label = min(Y_Train); 
i = 0; 
class = zeros(max(Y_Train),2);
for j= 1: max(Y_Train)
       class_temp = find( Y_Train == class_label); % account the number 
       i = i+1;  % i to account the number of class
       class(i,1)= class_label  ;     % the true lable of class i 
       class(i,2)= length(class_temp ); %the No.of elements in class i(labeled as class(i,1))
       class_label = class_label+1;   
end 
class(class(:,2)==0,:)=[];  % delete the etra row if the number of this class is zero. 
NoC = length(class);

%%
%------------------Get the estimation value of training data--------------
Mu = zeros(NoF,NoC);  %Number of Feature * Number of class
Sigma = zeros(NoF,NoF,NoC); %covariance matrix for NoC pages
P_w = zeros(NoC,1);   %Number of class * 1, storage the prior posibility of class i
classifier_failed=0;  %initial this classifier_failed
for i = class(:,1)'     %for class(1)
     Index_class = (Y_Train == i);  % all the index of ith class 
     X_Train_classi = (X_Train(:,Index_class))';  %all the data of ith class 
     % the output Mu(:,i), Sigma(:,:,i) are estimation value of training data of class i
     [Mu_temp, Sigma(:,:,i)] = GaussianMLEstimator(X_Train_classi);
     %%set a boundary for sigma matrix 
%      if det(Sigma(:,:,i))==0 || abs(det(Sigma(:,:,i)))< 1e-20
%           classifier_failed=classifier_failed+1;
%      else
     Mu(:,i)= Mu_temp';
     P_w(i)=  class(i,2)/ NoTrS; %the number of data of class i in trainng data 
%      end
     
end


%%
%----------------------- Put testing data into classifier------
% decide  wi(class i) 
%if        P(wi|x)>P(wj|x)          (for all j ~=i)
%=>  P(x|wi)*P(wi)>P(x|wj)*P(wj)     (for all j ~=i)

%introduce: gi(x)= ln(P(x|wi))+ln(P(wi))
%The final classifier is :
%    decide  wi(class i)  if  gi(x)>gj(x) (for all j ~=i)

%for Gaussian case,  we use log likelihood discriminant functions g(i)
if classifier_failed~=0
    Y_Test=zeros(NoTeS,1);
else
    g = zeros(NoC,1) ;
    warning('off','all')
    for Idx_test = 1:(NoTeS)     %for each data point in training sample
        %----------------------------------------------------------------------
        %-----Generate the classifier for class i ,g(i) is the distriminate----
        %funciton
        x = X_Test(:,Idx_test);  % NoF*1
        for i = 1:NoC       %Put in to the classifier of each class
          g(i)= -0.5* (x - Mu(:,i))'*(Sigma(:,:,i)^(-1))*(x - Mu(:,i))- NoF/2*log(2*pi)-0.5*log(det(Sigma(:,:,i)))+log(P_w(i));
    %       if g(i)<1e-19
    %           g(i)=0;
    %       end      
        end
       %-----------------------------------------------------------------------
      [maxg,Idx_maxg]= max(g);
    %   if maxg==0;
    %        Y_Test(Idx_test) = NaN;
    %   else      
           Y_Test(Idx_test) = Idx_maxg; % if gi(x)>gj(x) (for all j ~=i) classify this data as class i
    %   end
    end 
end    