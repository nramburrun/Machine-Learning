function [J, grad] = costFunctionReg(theta, X, y, lambda)
% Compute cost and gradient for logistic regression with regularization


% Initialize some useful values
m = length(y); % number of training examples
J = 0;
grad = ones(size(theta));

[J , grad] = costFunction(theta, X, y);
theta(1) = 0;
%sum_of_J = -y'*log(sigmoid(X*theta)) - (1-y')*(log(1-sigmoid(X*theta)));
J = J + (lambda/(2*m))*sum(theta.^2);

reg_grad = grad + (lambda/m)*theta;

% temp0 = grad(2:end) + reg_grad(2:end);
% grad(2:end) = temp0;

grad = reg_grad;

end
