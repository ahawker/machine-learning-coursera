function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %

    % Compute theta values via batch gradient descent.
    % Avoiding a loop here as we only have two theta parameters.
    t1 = theta(1) - alpha * (1 / m) * sum(((X * theta) - y) .* X(:, 1));
    t2 = theta(2) - alpha * (1 / m) * sum(((X * theta) - y) .* X(:, 2));

    % Feed our new theta values before computing cost.
	% Not done inline 
    theta(1) = t1;
    theta(2) = t2;

    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
