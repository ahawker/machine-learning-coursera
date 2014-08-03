function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Hint: The computation of the cost function and gradients can be
%       efficiently vectorized. For example, consider the computation
%
%           sigmoid(X * theta)
%
%       Each row of the resulting matrix will contain the value of the
%       prediction for that example. You can make use of this to vectorize
%       the cost function and gradient computations. 
%
% Hint: When computing the gradient of the regularized cost function, 
%       there're many possible vectorized solutions, but one solution
%       looks like:
%           grad = (unregularized gradient for logistic regression)
%           temp = theta; 
%           temp(1) = 0;   % because we don't add anything for j = 0  
%           grad = grad + YOUR_CODE_HERE (using the temp variable)
%

% NOTE:
% This is identical to my ex2 solution for a regularized cost
% function for logistical regression. I'm not sure why we are doing this twice?
% Possibly because submission from ex2 weren't required to be vectorized
% and this one has a large enough dataset to check the performance implications?

% identical to costFunction.m
h = sigmoid(X * theta);

% cost term - identical to costFunction.m
cost = (1 / m) * sum((-y .* log(h)) - ((1 - y) .* log(1 - (h))));

% regularize term - skipping theta(0)
% "end" keyword here refers to last element in matrix so we can subset the matrix
regCost = (lambda / (2 * m)) * norm(theta([2:end])) ^ 2;

% identical to costFunction.m
grad = (1 / m) .* X' * (h - y);

% regularize gradient term - skipping theta(0)
regGrad = (lambda / m) .* theta;
regGrad(1) = 0;

% Regularization just consists of summing the original cost
% with the regularized term.
J = cost + regCost;
grad = grad + regGrad;
% =============================================================

grad = grad(:);

end
