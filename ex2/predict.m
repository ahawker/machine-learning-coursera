function p = predict(theta, X)
%PREDICT Predict whether the label is 0 or 1 using learned logistic 
%regression parameters theta
%   p = PREDICT(theta, X) computes the predictions for X using a 
%   threshold at 0.5 (i.e., if sigmoid(theta'*x) >= 0.5, predict 1)

m = size(X, 1); % Number of training examples

% You need to return the following variables correctly
p = zeros(m, 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned logistic regression parameters. 
%               You should set p to a vector of 0's and 1's
%

% calculate our values
p = sigmoid(X * theta);

% Iterate through values, replacing them with 0 or 1 based on
% sigmoid calculated value.
% NOTE: Likely a non-iterative way to do this one, but this was too
% easy to pass up. :)
for i = 1:size(p),
	if (p(i) >= 0.5)
		p(i) = 1;
	else
		p(i) = 0;
	endif
end

% =========================================================================


end
