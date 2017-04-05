function acc = ComputeAccuracy(X, y, W, b)
K=10;
P=[];
n=size(X,2);
for i=1:n
    xp1=X(:,i);
    s =W*xp1+b;
    P1=softmax(s);
    P=[P,P1];
end

[argvalue, argmax] = max(sum(P));
acc=argmax/n;




% 
% each column of X corresponds to an image and X has size dn.
%  Y is the vector of ground truth labels of length n.
%  acc is a scalar value containing the accuracy.