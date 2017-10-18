
    function  Accuracy=AdhocAnalysis(ProjectedData,Y_Train,Y_Test_Desired,FeatureReductor,DimMax,Classifier,KMax)
        if (~exist('KMax','var'))   %there is just Dimension o Reduction to adhoc
                KMax = [];
                Accuracy=zeros(1,DimMax);

                maxacu=0;
                %for Dim=DimMax
                for Dim=1:DimMax
                    for i=1:5
                        if strcmp(ProjectedData(i).FeatureReductor,FeatureReductor)
                            m=i;  %the index of FeatureReductor
                        end
                    end            
                    X_Train_Proj= ProjectedData(m).ProjData(Dim). X_Train_Proj ;
                    X_Test_Proj = ProjectedData(m).ProjData(Dim). X_Test_Proj ;

                    Y_Test=Classification(X_Train_Proj, Y_Train, X_Test_Proj,Classifier);
                    Accuracy(Dim)=  sum(Y_Test_Desired==Y_Test)/ length(Y_Test);
                    if Accuracy(Dim)>maxacu;
                        maxacu=Accuracy(Dim);
                        Y_Test_Best=Y_Test;  %storage the best predict label when it reach the highest accuracy 
                        Dim_Best=Dim;
                    end        
                end 
                %generate the the accuracy plot
                figure(1)
               
                xaxis= 1:DimMax;
                plot(xaxis,Accuracy);grid on
                xlabel('Number of Projected Dimension'),ylabel('Accuracy')
                title({['The performance of ',FeatureReductor,'--',Classifier];['best on Dim=',num2str(Dim_Best),'  Highest Accuarcy=',num2str(maxacu*100),'%']})
                saveas(gcf,['./Results/',FeatureReductor,'--',Classifier,'[Accuracy Plot].fig'])
                
                close gcf;
                
        else    %there are Dim and Number of K to adhoc
                %Accuracy=zeros(KMax,DimMax);
                Accuracy=zeros(KMax);
                maxacu=0;
                for K=1:KMax
                    Dim=DimMax;
                   % for Dim=1:DimMax
                        for i=1:5
                            if strcmp(ProjectedData(i).FeatureReductor,FeatureReductor)
                                m=i;  %the index of FeatureReductor
                            end
                        end            
                        X_Train_Proj= ProjectedData(m).ProjData(Dim). X_Train_Proj ;
                        X_Test_Proj = ProjectedData(m).ProjData(Dim). X_Test_Proj ;

                        Y_Test=Classification(X_Train_Proj, Y_Train, X_Test_Proj,Classifier,K);
                        Accuracy(K)=  sum(Y_Test_Desired==Y_Test)/ length(Y_Test);
                        if Accuracy(K)>maxacu;
                            maxacu=Accuracy(K);
                            Y_Test_Best=Y_Test;  %storage the best predict label when it reach the highest accuracy 
                            K_Best=K;
                            Dim_Best=Dim;
                        end        
                   % end 
                end
                %generate the the accuracy plot
                
%                 figure(1)
%                 xaxis= 1:DimMax;
%                 for K=1:KMax
%                     plot(xaxis,Accuracy(K,:));hold on
%                     legendname(K,:)={strcat('K=',num2str(K))};
%                 end
%                 grid on;hold off
%                 legend(legendname')   
%                 xlabel('Number of Projected Dimension'),ylabel('Accuracy')
%                 title({['The performance of ',FeatureReductor,'--',Classifier];['best on Dim=',num2str(Dim_Best),'; K=',num2str(K_Best)];['  Highest Accuarcy=',num2str(maxacu*100),'%']})
%                 saveas(gcf,['./Results/',FeatureReductor,'--',Classifier,'(Accuracy Plot).fig'])
%                 close gcf;
        
        end
        
        %evaluation on the classification result for best dimension
        
        targets=ind2vec(Y_Test_Desired');
        outputs=ind2vec(Y_Test_Best');
        
        figure(2)
        plotconfusion(targets,outputs);
        saveas(gcf,['./Results/',FeatureReductor,'--',Classifier,'(Confusion on best Dim-)',num2str(Dim_Best),'.fig'])
        close gcf;
        plotroc(targets,outputs)
        saveas(gcf,['./Results/',FeatureReductor,'--',Classifier,'(ROC on best Dim-)',num2str(Dim_Best),'.fig'])
        close gcf;
    end