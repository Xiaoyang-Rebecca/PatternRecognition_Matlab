%write code for GaussianMLClassifier1.m which corresponds to “case 1” discussed in class
%(covariance is identical and a scaled identity matrix for each class). 


function [Y_Test] = GaussianMLClassifier1(X_Train, Y_Train, X_Test)
%Training procedure: using the training data to determine the parameters
%using the ML estimator for the Gaussian case.

[~  ,NoTrS] = size(X_Train); 
[NoF,NoTeS] = size(X_Test);
Y_Test=zeros(NoTeS,1);

%Assume that the data is labelled in continuous integer: 1,2,3...,Noc
%NoC = max(Y_Train);   %Number of class(label)

%%If the data is not labelled in continuous integer,but 0,2,5,7,8...
% we use class()
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
for i = class(:,1)'     %for class(1)
     Index_class = (Y_Train == i);  % all the index of ith class 
     X_Train_classi = (X_Train(:,Index_class))';  %all the data of ith class 
     % the output Mu(:,i), Sigma(:,:,i) are estimation value of training data of class i
     [Mu_temp, Sigma(:,:,i)] = GaussianMLEstimator(X_Train_classi);
%[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[Differenc to GaussianMLClassifier3]]]]]]]]]]]]]]]]]]] 
%learn the full covariance matrix, and then find out the "average standard deviation" 
     cov = mean(diag(Sigma(:,:,1)));
     
%[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[Differenc to GaussianMLClassifier3]]]]]]]]]]]]]]]]]]]      
     
     Mu(:,i)= Mu_temp';
     P_w(i)=  class(i,2)/ NoTrS; %the number of data of class i in trainng data 
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
g_C1 = zeros(NoC,1) ;
for Idx_test = 1:(NoTeS)     %for each data point in training sample
    %----------------------------------------------------------------------
    %-----Generate the classifier for class i ,g(i) is the distriminate----
    %funciton
    x = X_Test(:,Idx_test);  % NoF*1
    for i = class(:,1)'       %Put in to the classifier of each class
 %[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[Differenc to GaussianMLClassifier3]]]]]]]]]]]]]]]]]]] 
        w= Mu(:,i)./cov;
        w_0=-1/2*cov* Mu(:,i)'* Mu(:,i) + log(P_w(i));
        g_C1(i)= w'* x + w_0;
%[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[Differenc to GaussianMLClassifier3]]]]]]]]]]]]]]]]]]] 
    end
   %-----------------------------------------------------------------------
   Y_Test(Idx_test) = find(g_C1 == max(g_C1)); % if gi(x)>gj(x) (for all j ~=i) classify this data as class i
end
          