function [TestLabel] = KNNClassifier(TrainData, TrainLabel, TestData,K1)
 
 %Rows of X and Y correspond to observations and columns correspond to variables.
%%---------------------------------------
%%---------------------------------------
%%Copy right Xiaoyang LI ,University of Houston May,2016
 
 
 
[NoTrS  ,~] = size(TrainData); 
[NoTeS,~] = size(TestData);
TestLabel = zeros(NoTeS,1);

%%---------------------Get the class labels---------------------------
%Assume that the data is labelled in continuous integer: 1,2,3...,Noc
%NoC = max(TrainLabel);   %Number of class(label)

%If the data is not labelled in continuous integers,but 0,2,5,7,8...
% we use class(NoC:,2) to display the attribute of class
class_label = min(TrainLabel); 
i = 0; 
class = zeros(max(TrainLabel),2);
for j= 1: max(TrainLabel)
       class_temp = find( TrainLabel == class_label); % account the number 
       i = i+1;  % i to account the number of class
       class(i,1)= class_label  ;     % class(i,1) the true label of class i 
       class(i,2)= length(class_temp ); %the No.of elements in class i(labeled as class(i,1))
       class_label = class_label+1;   
end 
class(class(:,2)==0,:)=[];  % delete the etra row if the number of this class is zero. 
NoC = length(class);


%K1=2;  
	%Kn: the Kn nearest neighbors of x, estimate it by setting Kn= k1(n^0.5)
	K=K1*round(NoTrS^0.5);   % Larger values of k generalize better, and smaller values may tend to overfit.
	
	%[IDX,D] = knnsearch(X,Y,'k',K)  % X neighbors, Y centers
    %Rows of X and Y correspond to observations and columns correspond to variables.
	[KNNID_TrainData,~] = knnsearch(TrainData,TestData,'k',K);

	for j = 1:(NoTeS)     %for each data point in testing sample
		%----------------------------------------------------------------------
		%-----Generate the classifier for class i ,g(i) is the distriminate----
		%funciton
        KNNID_Testlabel = TrainLabel(KNNID_TrainData(j,:));

        ki=zeros(NoC,1);
		for i = 1:NoC       %Put in to the classifier of each class,ki is the number of neighbord 
		  %pn(wi|x)=pn(x|wi)/sumj(Pn(x|wj))= ki/k
		  ki(i)=sum(KNNID_Testlabel==class(i,1)); %class(i,1) the true label of class i 
		  g(i)= ki(i)/K;
		end
	   %-----------------------------------------------------------------------
	  [~,Idx_maxg]= max(g);
	   TestLabel(j) = Idx_maxg; % if gi(x)>gj(x) (for all j ~=i) classify this data as class i
	end
end




