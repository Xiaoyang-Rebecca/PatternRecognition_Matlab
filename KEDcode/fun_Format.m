function [sfrate] = fun_Format(srate,len)

% If you use this code, please cite the following paper.

% Reference:
% K. K. Huang, D. Q. Dai, C. X. Ren and Z. R. Lai, Learning Kernel Extended Dictionary for Face Recognition,
% IEEE Transation on Neural Network Learning System, vol. pp, no. pp, 1-13, 2016
% Email: kkcocoon@163.com (K. K. Huang)

if ~exist('len') len=4; end

sfrate = '';
for ri=1:length(srate)
    rate = srate(ri);
    frate = num2str(eval(vpa(rate,len)));
    if length(frate)==4
        frate = [frate, '0'];
    elseif length(frate)==3
        if rate==100
            frate = '100.0';
        else
            frate = [frate, '00'];
        end
    elseif length(frate)==2
        frate = [frate, '.00'];
    elseif length(frate)==1
        frate = ['' frate, '.000']; % ['0' frate, '.00'];
    end
    sfrate = [sfrate, frate,'  '];
end