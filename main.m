clear all
close all
addpath 'cifar-10-batches-mat';
%% Loading data

fprintf('1. Data read from the CIFAR-10 batch file');
[X1, Y1, y1] = LoadBatch('data_batch_1');  % batch 1.mat for training
[X2, Y2, y2] = LoadBatch('data_batch_2');  % batch 2.mat for validation
[X3, Y3, y3] = LoadBatch('test_batch');    % batch.mat for testing
fprintf(' - done \n');
%%  Setting values for GDparams and lambda
GD=GDparams;
GD.n_batch=100;
GD.n_epochs=40;
GD.eta=0.01;
lambda=0;
%%  Initialization of W and b
fprintf('2. Initialization of W and b');

K= 10; 
mean=0;
std=0.01;
%mean 0 standard variation 0.01
W = mean + std.*randn(K, 32,32,3);
b = mean + std.*randn(K, 1);
fprintf(' - done \n');
%%  Classifier and reshaping matrices for further purposes
fprintf('3. Evaluation of the network function');
n=length(X1);
xp=reshape(X1,3072,n);        %This is trainX in size 3072x10000
wp=reshape(W,K,3072);         %This is W in size 10x3072
P = EvaluateClassifier(xp, wp, b);
fprintf(' - done \n');
%% Cost calculation
fprintf('4. Calculation of the cost function');
J = ComputeCost(xp, Y1, wp, b, lambda);
fprintf(' - done \n');
%% Accuracy calculation
fprintf('5. Calculation of the accuracy of the network prediction');
acc = ComputeAccuracy(xp, Y1, wp, b);
fprintf(' - done \n');
%% Gradients calculation
fprintf('6. Gradients of the Cost Function');
[LW, Lb, ga] = ComputeGradients(xp, Y1, P, wp, lambda);
fprintf(' - done \n');
%%  Section for evaluating the gradients used in the report
% lamba=0;
% fprintf('running \n');
% [btest, wtest] = ComputeGradsNumSlow(xp(:,1:50), Y1(:,1:50),wp, b, lambda, 1e-6);
% fprintf('num done \n');
% [lwtest, lbtest, gatest] = ComputeGradients(xp(:,1:50), Y1(:,1:50), P, wp, lambda);
% fprintf('analyt done \n');
% difLW=[];
% for i=1:size(LW,1)
%     D1 = abs(lwtest(i,:)-wtest(i,:))/max(0,abs(lwtest(i,:)+abs(wtest(i,:))));
%     difLW = [difLW;D1];
% end
% difLB=[];
% for i=1:size(Lb,1)
%     D2 = abs(lbtest(i,:)-btest(i,:))/max(0,abs(lbtest(i,:)+abs(btest(i,:))));
%     difLB = [difLB;D2];
% end
%% MiniBatch 
fprintf('7. Mini-Batch Gradients');
[Wstar, bstar, JK1] = MiniBatchGD(xp, Y1, GD, wp, b, lambda);   %JK1 is an array with the cost value after every epoch
fprintf(' - done \n');
acc = ComputeAccuracy(xp, Y1, Wstar, bstar);
fprintf(['       -> Accuracy: ' strcat(num2str(acc),'%%')  '\n']);
%% Calculating functions for validated data
fprintf('8. Functions calculated for Validated Data');
xp2=reshape(X2,3072,n);                     %This is valX in size 3072x10000
Pval = EvaluateClassifier(xp2, wp, b);      %Classifier for Validated data
J2 = ComputeCost(xp2, Y2, wp, b, lambda);   %Cost for validated data
acc2 = ComputeAccuracy(xp2, Y2, wp, b);     %Accuracy for validated data
[LW2, Lb2] = ComputeGradients(xp2, Y2, Pval, wp, lambda);       %Gradients for validated data
[Wstar2, bstar2, JK2] = MiniBatchGD(xp2, Y2, GD, wp, b, lambda);  %MiniBatch for validated data - JK2 is an array with the cost value after every epoch
fprintf(' - done \n');
%% Comparison of loss between trained and validated data using: batch=100, eta=.01, n epochs=20 and lambda=0 
figure;
plot([1:GD.n_epochs],JK1,'b',[1:GD.n_epochs],JK2,'r');      %Plotting the loss function of training and validation data
legend('Traininig loss','Validation loss');
acc2 = ComputeAccuracy(xp2, Y2, Wstar2, bstar2);
% fprintf('9. Functions calculated for Validated Data \n');
% fprintf(['       -> Accuracy: ' strcat(num2str(acc),'%%')  '\n']);
%% Representation of the learnt weigth matrix
fprintf('10. Visualization of Weight  matrix W');
for i=1:10
im = reshape(Wstar(i, :), 32, 32, 3);
s_im1{i} = (im - min(im(:))) / (max(im(:)) - min(im(:)));
s_im1{i} = permute(s_im1{i}, [2, 1, 3]);
end
figure;
subplot(2,5,1);imshow(s_im1{1});
subplot(2,5,2);imshow(s_im1{2});
subplot(2,5,3);imshow(s_im1{3});
subplot(2,5,4);imshow(s_im1{4});
subplot(2,5,5);imshow(s_im1{5});
subplot(2,5,6);imshow(s_im1{6});
subplot(2,5,7);imshow(s_im1{7});
subplot(2,5,8);imshow(s_im1{8});
subplot(2,5,9);imshow(s_im1{9});
subplot(2,5,10);imshow(s_im1{10});
fprintf(' - done \n');
%% Evaluation of functions for the report, this section was used for 
%  presenting the loss of function in the report
%  by using different parameters for lambda and eta, while having a constant
%  number of epochs and batch

% lambda = [0,0,.1,1];
% gdt=GDparams;
% batch=100;
% epochs=40;
% eta=[.1,.01,.01,.01];
% 
% figure
% supt = suptitle('Comparison of loss between trained and validated data');
% set(supt,'FontSize',15);
% 
% for i=1:4
% gdt(i).n_batch=batch;
% gdt(i).eta=eta(i);
% gdt(i).n_epochs=epochs;
% [Wstar, bstar, JK1] = MiniBatchGD(xp, Y1, gdt(i), wp, b, lambda(i));
% acc = ComputeAccuracy(xp, Y1, Wstar, bstar);
% fprintf(['       -> Accuracy: ' strcat(num2str(acc),'%%')  '\n']);
% [Wstar, bstar, JK2] = MiniBatchGD(xp2, Y2, gdt(i), wp, b, lambda(i));
% subplot(2,2,i);
% plot([1:gdt(i).n_epochs],JK1,'b',[1:gdt(i).n_epochs],JK2,'r');
% xlabel(strcat(num2str(acc),'%%'));
% title(['lambda: ' num2str(lambda(i)) ' epochs: ' num2str(gdt(i).n_epochs) ' batch: ' num2str(gdt(i).n_batch) ' eta: ' num2str(gdt(i).eta) ]);
% end
% fprintf('11. Comparison of loss between trained and validated data using different parameters \n');
%%
fprintf('Code ran successfully \n\n');
