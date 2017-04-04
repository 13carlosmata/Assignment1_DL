function [X, Y, y] = LoadBatch(filename)

load('batches.meta.mat')
A = load (filename);
I = reshape(A.data',32,32,3,10000);
I = permute(I, [2,1,3,4]);
montage(I(:,:,:,:),'Size',[10,10]);
Y = zeros(10,10000,'uint8');
X = im2double(I);


for i=1:10000
   Y((A.labels(i)+1),i)=1;
end
y={};
for i=1:10000
   y{i}= label_names(A.labels(i)+1);
end
y=y';
end

