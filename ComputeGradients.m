function [LW, Lb,g] = ComputeGradients(X, Y, P, W, lambda)
n=size(X,2);
Lb=0;
LW=0;

for i=1:n
    %individuales
    Pi=P(:,i);
    Yt=(Y(:,i))';
    Xt=(X(:,i))';
  
    %ops
    g=-Yt/(Yt*Pi) * (diag(Pi)-Pi*(Pi)');
    Lb=Lb+g';
    lw=(g')*Xt;
    LW=LW+lw;  
end
LW=LW/n;
Lb=(Lb/n);


end


