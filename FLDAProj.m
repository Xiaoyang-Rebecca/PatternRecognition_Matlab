function [X_Train_Proj,X_Test_Proj] = FLDAProj(X_Train, Y_Train, X_Test, Dim)
	%Dim: dimensionality of the projected subspace 
    %Dim<= NoF
	[~  ,NoTrS] = size(X_Train); 
	[NoF,NoTeS] = size(X_Test);

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
       class(i,1)= class_label  ;       % [the true label of class i ]
       class(i,2)= length(class_temp ); % [the No.of elements in class i](labeled as class(i,1))
       class_label = class_label+1;   
end 
class(class(:,2)==0,:)=[];  % delete the etra row if the number of this class is zero. 
NoC = length(class(:,1));

%%
%------------------Get the estimation value of training data--------------
Mu = zeros(NoF,NoC);  %Number of Feature * Number of class

for i = class(:,1)'     %for class(1)
     Index_class = (Y_Train == i);  % all the index of ith class 
     X_Train_classi = (X_Train(:,Index_class));  %all the data of ith class 
     % the output Mu(:,i), Sigma(:,:,i) are estimation value of training data of class i
     Mu_temp= mean(X_Train_classi,2);
     Mu(:,i)= Mu_temp;           % [mean value of class(i)]
     %class(i,2) %the number of data of class i in trainng data 
end

 % FLDA 
     
    %Define the WITHIN class scatter matriX_Train
	Sw = 0;
    for c= class(:,1)' % for c classes
        s=zeros(NoF,NoF);
        for k = 1: NoTrS   % for d dimensions (features)
           s = s +(X_Train(:,k) - Mu(:,c)) *(X_Train(:,k) - Mu(:,c))';
        end
        Sw = Sw + s;
    end
    
    
    %Define the BETWEEN class scatter matriX_Train
    temp_sum = 0;
    for i = class(:,1)' % for c classes
        temp_sum = temp_sum + class(i,2)* Mu(:,i);  %ni=class(i,2)
    end
    Total_mean = temp_sum/ NoTrS;  %totoal mean vector
        
    Sb = 0;
	for c = class(:,1)' % for c classes
        Sb = Sb + class(c,2)*(Mu(:,c)-Total_mean)*(Mu(:,c)-Total_mean)';
    end
    
    S = (Sw)^(-1)*Sb; %calculate teh Eigen vectors and Eigen valuews of S
    [EigVec,EigVal] = eig(S) ;
    [~,ind_SortedEigVal] = sort(diag(EigVal),'descend');
	
	New_U = EigVec(:,ind_SortedEigVal(1:Dim));%select d Eigen Vectors corresponding to d highest Eigen values
	
    
	X_Train_Proj = (New_U'* X_Train)';
    
    X_Test_Proj  = (New_U'* X_Test)';
    
    
    
 end

