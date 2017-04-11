function J = ComputeCost(X, Y, W, b, lambda)
n=size(X,2);
lcross=0;
tr1=(Y)';
P = EvaluateClassifier(X, W, b);
for i=1:n
  lcross=-log(tr1(i,:)*P(:,i))+lcross;
end
J1=lcross/n;
sumW=sum(W.^2);
J2=lambda*sum(sumW);
J=J1+J2;
end