function P = EvaluateClassifier(X, W, b)
K=10;
P=[];
n=size(X,2);
for i=1:n
    xp1=X(:,i);
    s =W*xp1+b;
    P1=softmax(s);
    P=[P,P1];
end

 




