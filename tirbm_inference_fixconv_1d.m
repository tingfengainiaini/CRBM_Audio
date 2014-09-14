%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% ****************计算 卷积层卷积值以及激活值，**********%%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [poshidexp2 poshidprobs2] = tirbm_inference_fixconv_1d(imdata, W, hbias_vec,C_sigma,std_gaussian)

ws = size(W,1);
numbases = size(W,3);
numchannel = size(W,2);

poshidexp2 = zeros(size(imdata,1)-ws+1, 1, numbases);
if nargout>1
    poshidprobs2 = zeros(size(poshidprobs2));
end

% poshidexp2 = zeros(size(imdata,1)-ws+1, size(imdata,2)-ws+1, numbases);
for c=1:numchannel
    H = reshape(W(end:-1:1, c, :),[ws,1,numbases]); %%  对应到300片，每个channel的卷积核
    %% 下面是求80 个channel 的卷积之后的和啊！！！按时序进行卷积的，每个channel的卷积核不同，又因为初始化的poshidexp2初始化为0；
    %%  所以下面就相当于：80*1
    poshidexp2 = poshidexp2 + reshape(conv2_mult(imdata(:,:,c), H, 'valid'), size(poshidexp2));
end

for b=1:numbases
    %% 下面是逐个求300 个 group 
    poshidexp2(:,:,b) = C_sigma/(std_gaussian^2).*(poshidexp2(:,:,b) + hbias_vec(b));
    if nargout>1
        %% 这不是sigmoid激活嘛，在卷积神经网络里面是没有激活函数的，但是在RBM里是有的，这里虽然求了，但是貌似没有。
        poshidprobs2(:,:,b) = 1./(1 + exp(-poshidexp2(:,:,b))); 
    end
end
return