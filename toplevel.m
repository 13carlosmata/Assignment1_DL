addpath 'cifar-10-batches-mat';
addpath 'Datasets';
% batch 1.mat for training
% batch 2.mat for validation
% batch.mat for testing
[X1, Y1, y1] = LoadBatch('data_batch_1');
[X2, Y2, y2] = LoadBatch('data_batch_2');
[X3, Y3, y3] = LoadBatch('test_batch');
GD=GDparams;
GD.n_batch=100;
GD.n_epochs=20;
GD.eta=0.01;
%%
K= 10; 
mean=0;
std=0.01;
%mean 0 standard variation 0.01
W = mean + std.*randn(K, 32,32,3);
b = mean + std.*randn(K, 1);
%%
n=length(X1);
xp=reshape(X1,3072,n);
wp=reshape(W,K,3072);
P = EvaluateClassifier(xp, wp, b);
%P = EvaluateClassifier(xp(:, 1:100), wp, b);   %%Testing the function
%%
lambda=0;
J = ComputeCost(xp, Y1, wp, b, lambda);
%%
acc = ComputeAccuracy(xp, Y1, wp, b);
%%
[LW, Lb] = ComputeGradients(xp, Y1, P, wp, lambda);
%% [ngrad_b, ngrad_W] = ComputeGradsNumSlow(xp(:, 1), Y1(:, 1), wp, b, lambda, 1e-6);
%%
GD=GDparams;
[Wstar, bstar, JK1] = MiniBatchGD(xp, Y1, GDparams, wp, b, lambda);
%% For validated 
xp2=reshape(X2,3072,n);
Pval = EvaluateClassifier(xp2, wp, b);
J2 = ComputeCost(xp2, Y2, wp, b, lambda);
acc2 = ComputeAccuracy(xp2, Y2, wp, b);
[LW2, Lb2] = ComputeGradients(xp2, Y2, Pval, wp, lambda);
[Wstar, bstar, JK2] = MiniBatchGD(xp2, Y2, GDparams, wp, b, lambda);
%%

figure;
plot([1:GD.n_epochs],JK1,'b',[1:GD.n_epochs],JK2,'r');
legend('Trained','Validated');
acc = ComputeAccuracy(xp, Y1, Wstar, bstar);
