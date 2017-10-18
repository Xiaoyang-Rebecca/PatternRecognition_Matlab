function pdict_label = GMMClassifier( TrainSample, TrainLabel,TestSample, K )  
%input:
%       TrainSample:N-by-D data matrix.  
%       TrainLabel :N-by-1 data matrix.  
%       TestSample :M-by-D data matrix. 
%       K: number of component in each class(assumber every class have same number of component)
%Output:
%       pdict_label:N-by-D data matrix.  
%%---------------------------------------
%%---------------------------------------
%%Copy right Xiaoyang LI ,University of Houston May,2016


[NoTeS,NoF]=size(TestSample);
pdict_label=zeros(NoTeS,1);
%%
%fit GMM to trainsample,generate parameter for each clusters in each class
for i= unique(TrainLabel)' %i index of class

    X= TrainSample((TrainLabel==i),:);
    %for Samples in each class, implement GMM to get parameters
    
    MODEL=gmm(X,K); 
        
    field.para(i)=MODEL;
    %  - PX: N-by-K matrix indicating the probability of each
    %       component generating each point.(responsibility matrix)
    %  [~, cls_ind] = max(px,[],2); %cls_ind = cluster label  
    %  - MODEL: a structure containing the parameters for a GMM:
    %       MODEL.Miu: a D by K matrix.
    %       MODEL.Sigma: a D-by-D-by-K matrix.
    %       MODEL.Pi: a 1-by-K vector. (Posterior probability of each component  )
end    
NoClass=i;
    

%%
%predict labels for testing samples

%Method 1  Maximum Likelihood
    for Idx_test = 1:NoTeS     %for each data point in training sample
            %----------------------------------------------------------------------
            %-----Generate the classifier for class i ,g(i) is the distriminate----
            %funciton
          x = TestSample(Idx_test,:)';  % NoF*1
          g=zeros(NoClass,K);
          
          classlabel=0;
          maxg=-inf;
          for i= 1: NoClass%i index of class
              for j=1:K
                  Mu   =field.para(i).Miu; 
                  Sigma=field.para(i).Sigma;% a D-by-D-by-K matrix.   
                  P_w  = field.para(i).Pi;   % : a 1-by-K vector. (Posterior probability of each component  )
                  % sum up the posibility of all components in this class
                  g(i,j)= (-0.5* (x - Mu(:,j))'*(Sigma(:,:,j)^(-1))*(x - Mu(:,j))- NoF/2*log(2*pi)-0.5*log(det(Sigma(:,:,j)))+log(P_w(j)));
                  %g(i,j)= log(mvnpdf(x,Mu(:,j),(Sigma(:,:,j)))+log(P_w(j)));
                  if g(i,j)>maxg
                    maxg=g(i,j);
                    classlabel=i;
                  end
              end
          end
          pdict_label(Idx_test) = classlabel;
    end 
 %Method 2 KNN
 
end  


