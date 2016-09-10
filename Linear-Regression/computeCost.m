function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

m = length(y); % number of training examples

J = 0;
sum = 0;


for m_it = 1:m
       sum = sum + (theta(1) + theta(2)*(X(m_it,2))  - y(m_it))^2;
end
 J = (1/(2*m) ) * sum;

end
