function [reco_ratio,tt_ID] = fun_NN(TrainX,TrainClass,TestX,TestClass)

tr_num     =    size(TrainX,2);
tt_num     =    size(TestX,2);
tt_ID      =    zeros(1,tt_num);


tt_ID   = zeros(1,tt_num);
tic;
for i  =  1: tt_num
    gap        = zeros(1,tr_num);
    for j   =  1:tr_num
        gap(j) =  norm(TestX(:,i)-TrainX(:,j)); 
    end
    
    [mg,mi] = min(gap);
    tt_ID(i)  =  mi;
end
% toc;

reco_ratio=(sum(TrainClass(tt_ID)==TestClass(1:tt_num)))/tt_num; % output the recognition rate
reco_ratio = round(10000*reco_ratio)/100;