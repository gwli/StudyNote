function [W] = pca(X, k)
 [n,d] = size(X);
 mu = mean(X);
  for  i =1:size(X,2)
    NewX(:,i)=X(:,i)-mean(X(:,i));
    CovNewX  =NewX(:,i)'*NewX(:,i);
    Xm(:,i) = NewX(:,i)./sqrt(CovNewX);
end
  if(n>d)
    C = Xm'*Xm;
    [W,D] = eig(C);
    % sort eigenvalues and eigenvectors
    [D, i] = sort(diag(D), 'descend');
    W = W(:,i);
    % keep k components
    W = W(:,1:k);
  else
    C = Xm*Xm';
    %C = cov(Xm');
    [W,D] = eig(C);
    % multiply with data matrix
    W = Xm'*W;
    % normalize eigenvectors
    for i=1:n
      W(:,i) = W(:,i)/norm(W(:,i));
    end
    % sort eigenvalues and eigenvectors
    [D, i] = sort(diag(D), 'descend');
    W = W(:,i);
    % keep k components
    W = W(:,1:k);
  end
sum(D(1:k))/sum(D)