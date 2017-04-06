function [LW, Lb] = ComputeGradients(X, Y, P, W, lambda)
n=size(X,2);
Lb=0;
LW=0;
for i=1:n
    %individuales
    Pi=P(:,i);
    Yt=transpose(Y(:,i));
    Xt=transpose(X(:,i));
    %ops
    g=-((Yt*Pi)\Yt) * (diag(Pi)-Pi*transpose(Pi));
    Lb=Lb+g;
    lw=transpose(g)*Xt;
    LW=LW+lw;
end
LW=LW/n;
Lb=transpose(Lb/n);


end


