function J = ComputeCost(X, Y, W, b, lambda)
K=10;
P=[];
n=size(X,2);
D=n;
lcross=0;
tr1=transpose(Y);
for i=1:n
  for j=1:K
      if tr1(i,j)==1
          index=j;
      end
  end
  xp1=X(:,i);
  s =W*xp1 + b;
  P1=softmax(s);
  P=[P,P1];
  lcross=-log10(tr1(i,index)*P1(index))+lcross;
end
J1=lcross/10000;
sumW=sum(W(:));
J2=lambda*sumW^2;
J=J1+J2;
end