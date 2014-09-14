%% 参数设置。
function subtract_fea()

clear;
clc;

ws = 6;
spacing = 3;
C_sigma = 1;
std_gaussian = 0.2;
nPC = 80;
sigmaPC = 3;

%%  从训练模型提取特征。
data = load('Pall.mat');
Pall = data.Pall;
par = load('tirbm_audio_LB_TIMIT_V1b_w6_b300_pc80_sigpc3_p0.05_pl0.05_plambda5_sp3_exp_eps0.01_epsdecay0.01_l2reg0.01_bs01_20140904T230856.mat');
W = par.W;
hbias_vec = par.hbias_vec;
%% PCA + Whiting.
% 2. Run PCA
[X startframe_list] = concatenate_speech_data(Pall, randsample(length(Pall), min(length(Pall), 1000)));
% X = X-repmat(mean(X,2), 1, size(X,2)); % don't subtract mean
Cov = X*X'/size(X,2);               
[V D] = eig(Cov);                   

% 3. Try to reconstruct the original data with PCA components
numfeat = size(X,1);
E = V(:, numfeat:-1:numfeat-nPC+1);
S = diag(subvec(diag(D), numfeat:-1:numfeat-nPC+1));
Xpc = E'*X;
Xrec = E*Xpc;

Ewhiten = diag(sqrt(diag(S)+sigmaPC).^-1)*E';
Eunwhiten = E*diag(sqrt(diag(S)+sigmaPC));
Xrec2 = Eunwhiten*Ewhiten*X;

% visualization
figure, subplot(2,1,1), imagesc(X(:, 1:500)), 
subplot(2,1,2), imagesc(Xrec(:, 1:500))
figure, imagesc(Xrec2(:, 1:500));
figure, imagesc(E);

%%  
imdata = Ewhiten*Pall{ceil(rand()*length(Pall))};
imdata = trim_audio_for_spacing_fixconv(imdata, ws, spacing);       
imdatatr = imdata';
imdatatr = reshape(imdatatr, [size(imdatatr,1), 1, size(imdatatr,2)]);
conhidexp = tirbm_inference_fixconv_1d(imdatatr,W, hbias_vec,C_sigma,std_gaussian);
[conhidstates conhidprobs] = tirbm_sample_multrand_1d(conhidexp, spacing);       
%%  
end 