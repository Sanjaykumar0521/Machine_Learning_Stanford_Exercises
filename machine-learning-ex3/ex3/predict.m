function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);
%display(size(X));
X=[ones(m,1) X];
%display(size(X));
% You need to return the following variables correctly 
%p = zeros(size(X, 1), 1);
%display(size(X));
% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

zTwo=Theta1*X';
aTwo=sigmoid(zTwo);
%display(size(aTwo));
aTwo=[ones(size(aTwo'),1) aTwo'];
%display(size(aTwo));
%display(size(Theta2));
zThree=aTwo*Theta2';

aThree=sigmoid(zThree);
[v p]=max(aThree,[],2);







% =========================================================================


end
