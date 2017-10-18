function [X_Test_ds,Y_Test_Desired_ds]= downsample(X_Test,Y_Test_Desired,NoS)
    %NoS=10;   %number of samples perclass
    [class_label,index_perclass,~]= unique(Y_Test_Desired);
    Samp_Index_perclass=zeros(length(class_label),NoS);
    for i= class_label'
        if i== length(class_label)
             Samp_Index_perclass(i,:) = index_perclass(i)+randperm((length(Y_Test_Desired)-index_perclass(i)),NoS); 
        else
             Samp_Index_perclass(i,:) = index_perclass(i)+randperm((index_perclass(i+1)-index_perclass(i)),NoS); 
        end
    end
    Samp_Index= Samp_Index_perclass(:);

    X_Test_ds          = X_Test(:,Samp_Index);      %12197*NoF
    Y_Test_Desired_ds  = Y_Test_Desired(Samp_Index);   %the desire output of target(class value)
end