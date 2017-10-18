function pTrainOcc = fun_KED_dict(Xtr,Tro,Trc,W_kda,sigma,feaType,num)

% The key function to get KED.

% Reference:
% K. K. Huang, D. Q. Dai, C. X. Ren and Z. R. Lai, Learning Kernel Extended Dictionary for Face Recognition,
% IEEE Transation on Neural Network Learning System, vol. pp, no. pp, 1-13, 2016
% Email: kkcocoon@163.com (K. K. Huang)

Doo = fun_FeaDist(Tro,Tro,feaType);
Doc = fun_FeaDist(Tro,Trc,feaType);
Dco = Doc'; % fun_FeaDist(Trc,Tro,feaType);
Dcc = fun_FeaDist(Trc,Trc,feaType);
Dp = fun_FeaDist(Xtr,Tro,feaType);
Dpp = fun_FeaDist(Xtr,Trc,feaType);

Koo = exp(-Doo./sigma^2);
Koc = exp(-Doc./sigma^2);
Kco = Koc'; % exp(-Dco./tGrad^2);
Kcc = exp(-Dcc./sigma^2);
Kp = exp(-Dp./sigma^2);
Kpp = exp(-Dpp./sigma^2);

Ko = Koo - Koc - Kco + Kcc;

options = [];   
options.Kernel = 1;
[Vo, Do] = KPCA(Ko,options);
Vo = Vo(:,1:num);

% KDA projection
pTrainOcc = W_kda'*Kp*Vo - W_kda'*Kpp*Vo;