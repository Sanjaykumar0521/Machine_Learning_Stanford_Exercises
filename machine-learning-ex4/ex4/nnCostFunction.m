function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%


%#########   MY CODE   #########
%display(size(X));
%X=[ones(m,1) X];
%display(size(X));
%display(size(Theta1));
%z2=Theta1*X';
%a2=sigmoid(z2);
%display(size(a2));
%printf("a2 is ");
%a2=[ones(size(a2'),1) a2'];
%display(size(a2));
%display(size(Theta2));
%z3=Theta2*a2';
%display(size(z3));
%printf(" a3 is ");
%a3=sigmoid(z3)';
%display(size(a3));
%s=0;
%y
%display(size(y));
%y_m = eye(num_labels)(y,:);
%display(size(y_m)); 
%display(size(y_m));

%J=(-1/m)*sum(sum(y_m*log(a3)+(1-y_m)*log(1-a3)));
%for i=1:m
 %   J=J+(y_m(i)*log(a3)+(1-y_m(i))*log(1-a3));
%end;

%J=(1/m)*J;


%######## CLEAN CODE   ################
Delta1=0;
Delta2=0;


a1=[ones(m,1) X];
z2=Theta1*a1';
a2=sigmoid(z2);
a2=[ones(size(a2'),1) a2'];
z3=Theta2*a2';
a3=sigmoid(z3)';
y_m = eye(num_labels)(y,:);
%d3=a3-y_m;
%d2=Theta2(:,2:end)'*d3'.*sigmoidGradient(z2);
%display(size(d2));
%d2=d2(2:end);
%for i=1:m 
%  a1=
%  Delta1=a1*d2;
%  Delta2=a2'*d3;

%Theta1_grad=(1/m)*Delta1;
%Theta2_grad=(1/m)*Delta2;
%###### REGULARIZATION  #######
reg1=0;
reg2=0;
theta1=Theta1(:,2:end);
theta2=Theta2(:,2:end);
%display(size(Theta1));
%display(size(Theta2));

%######    USING VECTORIZATION    #########
reg1=sum(sum(theta1.^2));
reg2=sum(sum(theta2.^2));

%######    USING LOOP    #########
%for i=1:hidden_layer_size
%  for j=1:input_layer_size
%    reg1=reg1+(Theta1(i,j)^2);
%  end
%end

%for i=1:num_labels
%  for j=1:hidden_layer_size
%    reg2=reg2+(Theta2(i,j)^2);
%  end
%end

J=(1/m)*sum(sum((-y_m.*log(a3))-(1-y_m).*log(1-a3)))+(lambda/(2*m))*(reg1+reg2);
%display(size(a3));
%display(size(y));
%display(size(sigmoidGradient(z2)));


  a1=[ones(m,1) X];
  z2=Theta1*a1';
  a2=sigmoid(z2);
  a2=[ones(size(a2'),1) a2'];
  z3=Theta2*a2';
  a3=sigmoid(z3)';
  d3=a3-y_m;
  
  %display(size(theta2));
  %display(size(d3));
  d2=theta2'*d3'.*sigmoidGradient(z2);
  %display(size(d2));
  %d2=d2(:,2:end);
  %display(size(d2));
  %display(size(a1));
  Delta1=d2*a1;
 % display(size(d3));
  %display(size(a2));
 Delta2=d3'*a2;
 
  %display(size(Delta2));
  %display(size(Delta1));
  %display(size(Theta1));
  %display(size(Theta2));
  Theta1(:,1)=0;
  Theta2(:,1)=0;
Theta1_grad=(1/m)*Delta1+(lambda/m)*Theta1;
Theta2_grad=(1/m)*Delta2+(lambda/m)*Theta2;  
%display(size(Theta1_grad)); 
%firstTheta=0;
%secondTheta=0;
%for i=1:hidden_layer_size
%   for j=2:input_layer_size 
%        firstTheta=firstTheta+(lambda/m)*Theta1(i,j);
% end
%end
%for i=1:num_labels
%   for j=2:hidden_layer_size 
%         secondTheta=secondTheta+(lambda/m)*Theta2(i,j);
% end
%end

%Theta1_grad(2:end,:)=(1/m)*Delta1(2:end,:)+firstTheta;
%Theta2_grad(2:end,:)=(1/m)*Delta2(2:end,:)+secondTheta;





% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
