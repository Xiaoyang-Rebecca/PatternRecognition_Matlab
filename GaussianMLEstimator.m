%Copyright: Xiaoyang LI
%[Mu, Sigma] = GaussianMLEstimator(X)
%Input:  X=n*f  (Number of Samples * Number of features )
%Output: [Mu, Sigma] are maximun likelihood estimaition of mean and
%                     covariance of X, respectively
function [Mu, Sigma] = GaussianMLEstimator(X)

    [NoS,NoF] =size(X);
    %Estimate Mu of X
    sum_X = zeros(1,NoF); %initialize the summation

    for i = 1:NoS
        sum_X = sum_X + X(i,:);
    end
    Mu = sum_X ./ NoS;

    %Estimate Sigma of X
    sum_X = zeros(NoF,NoF); %initialize the summation
    for i = 1:NoS
        sum_X = sum_X + (X(i,:)-Mu)'*(X(i,:)-Mu);
    end
    Sigma = sum_X ./ NoS;

    
   




    
