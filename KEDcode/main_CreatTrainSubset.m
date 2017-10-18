
%% -------------------------------------------------------------------------
% randomly select training samples

% If you use this code, please cite the following paper.

% Reference:
% K. K. Huang, D. Q. Dai, C. X. Ren and Z. R. Lai, Learning Kernel Extended Dictionary for Face Recognition,
% IEEE Transation on Neural Network Learning System, vol. pp, no. pp, 1-13, 2016
% Email: kkcocoon@163.com (K. K. Huang)

clc;clear;close all;
load('db_PEAL');

XgalClass = TrainClass(ind0_gal);

sind=[];

for ri=1:3
    if ri==1
        ind_explig = [ind0_lig]; 
    elseif ri==2
        ind_explig = [ind0_exp];
    else
        ind_explig = [ind0_acc];
    end
    
    XexpligClass = TrainClass(ind_explig);
    classids = unique(XexpligClass);
    NumClass = length(classids);
    ind_peo = randperm(NumClass);
    
    for i=1:NumClass
        j = ind_peo(i);
        ind = find(XexpligClass==classids(j));
        indL = length(ind);
        
        if indL<4
            continue;
        end
        tii = randperm(indL);
        tii = sort(tii(1:4));
        
        if ri==3
            sind = [sind,ind(3:end)];
        else
            sind = [sind,ind(tii)];
        end
        
        if ri==1 && length(sind)/4>=200
            break;
        elseif ri==2 && length(sind)/4>=300
            break;
        elseif ri==3 && length(sind)/4>=320
            break;
        end
    end
end

ind_lig=ind0_lig; ind_lig(sind(1:200*4)) = [];
ind_exp=ind0_exp; ind_exp(sind(200*4+1:300*4)) = [];
ind_acc=ind0_acc; ind_acc(sind(300*4+1:end)) = [];

ind_tr = [ind0_lig(sind(1:200*4)),ind0_exp(sind(200*4+1:300*4))];
ind_tr_acc = ind0_acc(sind(300*4+1:end));

% adding normal sample
classids = unique(TrainClass([ind_tr]));
NumClass = length(classids);
for i=1:NumClass
    i1 = find(XgalClass==classids(i));
    ind_tr = [ind0_gal(i1),ind_tr];
end


%% occluding sample and its corresponding normal sample
PairOcc = [];
imageList_tr = imageList(ind_tr_acc);
class_tr = TrainClass(ind_tr_acc);
class_gal = TrainClass(ind0_gal);
for i=1:length(imageList_tr)
    s = imageList_tr{i};
    ind = strfind(s,'_A');
    if str2num(s(ind+2))>=4  % 6 kinds of occlusion 
          j = find(class_gal==class_tr(i));
          PairOcc = [PairOcc; ind0_gal(j), ind_tr_acc(i), 1];
    elseif str2num(s(ind+2))>=1 
          j = find(class_gal==class_tr(i));
          PairOcc = [PairOcc; ind0_gal(j), ind_tr_acc(i), 0];
    end
end

% testing
i=5;
a1 = reshape(TrainX(:,PairOcc(i,1)),im_h,im_w);
a2 = reshape(TrainX(:,PairOcc(i,2)),im_h,im_w);
figure;imshow(a1);
figure;imshow(a2);

save('db_PEAL_ind_tr_2','ind_tr','ind_tr_acc','ind_lig','ind_exp','ind_acc','PairOcc');  

