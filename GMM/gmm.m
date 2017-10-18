function model = gmm(X, K)
% ============================================================
% Expectation-Maximization iteration implementation of
% Gaussian Mixture Model.
%
% PX = GMM(X, K_OR_CENTROIDS)
% [PX MODEL] = GMM(X, K_OR_CENTROIDS)
%
%  - X: N-by-D data matrix.
%  - K: either K indicating the number of
%       components 
%
%  - PX: N-by-K matrix indicating the probability of each
%       component generating each oint.
%  - MODEL: a structure containing the parameters for a GMM:
%       MODEL.Miu: a K-by-D matrix.
%       MODEL.Sigma: a D-by-D-by-K matrix.
%       MODEL.Pi: a 1-by-K vector.
% ============================================================
% @SourceCode Author: Pluskid (http://blog.pluskid.org)
% @Appended by : Sophia_qing (http://blog.csdn.net/abcjennifer)
    

%% Generate Initial Centroids
    threshold = 1e-14;
    
    tol = 1e-6;
    maxiter = 500;
    llh = -inf(1,maxiter);

    [N, D] = size(X);
 
        
        %K-by-D matrix indicating the choosing of the initial K centroids.
        %we use kmeans to extract the centroids
        [~,centroids] = kmeans(X,K);
        
    %% initial values
    [pMiu,pPi,pSigma] = init_params();
 
    Lprev = -inf; 
    
    %% EM Algorithm
    for iter = 2:maxiter
        %% Estimation Step
        Px = real(calc_prob());     % size=Px(N, K);
        Px(isnan(Px))=0;
        Px(isnan(Px))=0;
       
        % new value for pGamma(N*k), pGamma(i,k) =the posibility that Xi
        % gengernate from Kth Gaussian Cluster
      
        pGamma = Px .* repmat(pPi, N, 1); %numerator  = pi(k) * N(xi | pMiu(k), pSigma(k))
        pGamma = pGamma ./ repmat(sum(pGamma, 2), 1, K); %denominator = pi(j) * N(xi | pMiu(j), pSigma(j))sum over j
        
        %% Maximization Step - through Maximize likelihood Estimation
        
        Nk = sum(pGamma, 1);    %number of samples in each cluster
        Nk(isnan(Nk))=0;
        Nk(isinf(Nk))=0;
        
        % update pMiu
        pMiu = diag(1./Nk) * pGamma' * X; %update pMiu through MLE
        pPi = Nk/N;
        
        
        % update k? pSigma
        for kk = 1:K 
            Xshift = X-repmat(pMiu(kk, :), N, 1);
            pSigma(:, :, kk) = (Xshift' * ...
                (diag(pGamma(:, kk)) * Xshift)) / Nk(kk);
        end
 
        % check for convergence
        L = sum(log(Px*pPi'));
        if L-Lprev < threshold || abs(llh(iter)-llh(iter-1)) < tol*abs(llh(iter))
            break;
        end
        Lprev = L;
        
    end
 
 
        model = [];
        model.Miu = pMiu';
        model.Sigma = pSigma;
        model.Pi = pPi;
 

    %% Function Definition
    
    function [pMiu,pPi,pSigma] = init_params()
        pMiu = centroids; 
        pPi = zeros(1, K); 
        pSigma = zeros(D, D, K); 
 
       
        distmat = repmat(sum(X.*X, 2), 1, K) + ... 
            repmat(sum(pMiu.*pMiu, 2)', N, 1) - ...
            2*X*pMiu';
        [~, labels] = min(distmat, [], 2);%Return the minimum from each row
 
        for k=1:K
            Xk = X(labels == k, :);
            pPi(k) = size(Xk, 1)/N;
            pSigma(:, :, k) = cov(Xk);
        end
    end
 
    function Px = calc_prob() 
        %Gaussian posterior probability 
        %N(x|pMiu,pSigma) = 1/((2pi)^(D/2))*(1/(abs(sigma))^0.5)*exp(-1/2*(x-pMiu)'pSigma^(-1)*(x-pMiu))
        Px = zeros(N, K);
        for k = 1:K
            Xshift = X-repmat(pMiu(k, :), N, 1); %X-pMiu
            inv_pSigma = inv(pSigma(:, :, k));
            tmp = sum((Xshift*inv_pSigma) .* Xshift, 2);
            coef = (2*pi)^(-D/2) * sqrt(det(inv_pSigma));
            Px(:, k) = coef * exp(-0.5*tmp);
        end
    end
end
