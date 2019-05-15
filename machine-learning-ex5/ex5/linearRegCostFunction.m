function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

%theta1=theta(2:end,:);
%display(size(X));
%display(size(y));
%display(size(theta));
%display(size(theta1));
[r c]=size(X);
%display(r);
%display(c);

h=X*theta;
thetaSqrSum=0;
thetaSum=0;
for i=2:c
  thetaSqrSum=thetaSqrSum+theta(i)^2;
  %thetaSum=thetaSum+theta(i);
end
%X(:,2:end)
%size(X)
J=(1/(2*m))*sum((h-y).^2)+(lambda/(2*m))*thetaSqrSum;
theta(1)=0;
grad=(1/m)*(X'*(h-y))+(lambda/m)*theta;
grad(1)=(1/m)*(X(:,1)'*(h-y));
%display(size(grad));







% =========================================================================

grad = grad(:);

end
