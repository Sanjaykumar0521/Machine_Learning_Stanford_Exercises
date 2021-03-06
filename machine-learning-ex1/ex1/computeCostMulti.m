function J = computeCostMulti(X, y, theta)
%COMPUTECOSTMULTI Compute cost for linear regression with multiple variables
%   J = COMPUTECOSTMULTI(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples
%prediction=X*theta;
% You need to return the following variables correctly 
%sqr=(prediction-y).^2;

%J=(1/(2*m))*sum(sqr);

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.
%X=X';
%display(X(1:10,:));
%display(y(1:10,:));
%display(theta);
%display(X*theta-y);
J=(1/(2*m))*(X*theta-y)'*(X*theta-y);
%J=(1/(2*m))*sum((X*theta-y).^2);

% =========================================================================

end
