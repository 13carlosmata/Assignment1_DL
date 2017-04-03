addpath 'cifar-10-batches-mat';
addpath 'Datasets';
% batch 1.mat for training
% batch 2.mat for validation
% batch.mat for testing
[X1, Y1, y1] = LoadBatch('data_batch_1');
[X2, Y2, y2] = LoadBatch('data_batch_2');
[X3, Y3, y3] = LoadBatch('test_batch');

%%
K= 10; 
mean=0;
std=0.01;
%mean 0 standard variation 0.01
W = mean + std.*randn(K, 32,32,3);
b = mean + std.*randn(K, 1);
%%
