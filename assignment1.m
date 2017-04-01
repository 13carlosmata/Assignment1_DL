i
%% Backgroud 3
addpath 'cifar-10-batches-mat';
addpath 'Datasets';
load('batches.meta.mat')

A = load ('data_batch_1.mat');
I = reshape(A.data',32,32,3,10000);
I = permute(I, [2,1,3,4]);
montage(I(:,:,:,:),'Size',[10,10]);
Y = zeros(10,10000,'uint8');
I2 = im2single(I);
X=I2/255;

for i=1:10000
   Y(i,(A.labels(i)+1))=1;
end
y={};
for i=1:10000
   y{i}= label_names(A.labels(i)+1);
end
y=y';

% batch 1.mat for training
% batch 2.mat for validation
% test batch.mat for testing
%%   Exercise 1 - Training a multi-linear classifier 

