function [Wstar, bstar, JK] = MiniBatchGD(X, Y, GDparams, W, b, lambda)
GD=GDparams;
N=size(X,2);
JK=[];
for i=1:GD.n_epochs
    for j=1:N/GD.n_batch
        %Composicion de los batches - del pdf
        j_start = (j-1)*GD.n_batch + 1;
        j_end = j*GD.n_batch;
        inds = j_start:j_end;
        Xbatch = X(:, j_start:j_end);
        Ybatch = Y(:, j_start:j_end);
        % calif del batch
        P = EvaluateClassifier(Xbatch, W, b);
        [LW, Lb] = ComputeGradients(Xbatch, Ybatch, P, W, lambda);
        W = W - GD.eta*LW;
        b = b - GD.eta*Lb;
    end
    J = ComputeCost(X, Y, W, b, lambda);
    JK = [JK;J];
    Wstar=W;
    bstar=b;
end
%plot(JK);
%JK
end
