function plotData(X, y)
%PLOTDATA Plots the data points X and y into a new figure 
%   PLOTDATA(x,y) plots the data points with + for the positive examples
%   and o for the negative examples. X is assumed to be a Mx2 matrix.

% Create New Figure
figure; hold on;

% ====================== YOUR CODE HERE ======================
% Instructions: Plot the positive and negative examples on a
%               2D plot, using the option 'k+' for the positive
%               examples and 'ko' for the negative examples.
%
%########## CODE WRITTEN BY ME  ############
postv=find(y==1);
negtv=find(y==0);

plot(X(postv,1),X(postv,2),'k+','LineWidth',2,'MarkerSize',10)
plot(X(negtv,1),X(negtv,2),'ko','MarkerFaceColor','y','MarkerSize',10)






% =========================================================================



hold off;

end
