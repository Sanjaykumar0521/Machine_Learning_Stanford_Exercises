function [X_norm, mu, sigma] = featureNormalize(X)
%FEATURENORMALIZE Normalizes the features in X 
%   FEATURENORMALIZE(X) returns a normalized version of X where
%   the mean value of each feature is 0 and the standard deviation
%   is 1. This is often a good preprocessing step to do when
%   working with learning algorithms.

% You need to set these values correctly
%X_norm=X;
%mu = mean(X(:,1));
%X_norm=X_norm-mu;
%sigma = std(X(:,1));
%X_norm=X_(norm)*(1/sigma);
% ====================== YOUR CODE HERE ======================
% Instructions: First, for each feature dimension, compute the mean
%               of the feature and subtract it from the dataset,
%               storing the mean value in mu. Next, compute the 
%               standard deviation of each feature and divide
%               each feature by it's standard deviation, storing
%               the standard deviation in sigma. 
%
%               Note that X is a matrix where each column is a 
%               feature and each row is an example. You need 
%               to perform the normalization separately for 
%               each feature. 
%
% Hint: You might find the 'mean' and 'std' functions useful.
%       
 mu=mean(X);
 sigma=std(X);
 %display(X(1:10,:));
 %display(mu(1));
 %display(mu);
 %display(X);
 %display(sigma);
 %r=std(X(:,1));
 %display(r);
 %display(mean(X(:,1)));
 X=X-mu;
 %display(X(1:10,:));
 %X=(X).*(1/sigma);
  X(:,1)=(X(:,1)/sigma(:,1));
  X(:,2)=(X(:,2)/sigma(:,2));
 %display(X(1:10,:));
 %X_norm=X;
 %display(X_norm(1:10,:));
 roundn = @(x,n) round(x*10^n)./10^n;
 X_norm = roundn(X,2);
 %X_norm=X;
 
 %display(X_norm(1:10,:));


% ============================================================

end
