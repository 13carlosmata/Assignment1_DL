function P = EvaluateClassifier(X, W, b)
P = zeros(K,10000);

for i=1:10000
   x_partial=X(:,:,:,i);
   w_partial=W(1,:,:,:);
   w_partial = permute (w_partial,[2,3,4,1]);
   s =  w_partial.*x_partial;
end




%p = softmax(s);





