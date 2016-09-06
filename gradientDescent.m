function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

m = length(y); 
J_history = zeros(num_iters, 1);
sum_theta_one=0;
sum_theta_zero=0;

for iter = 1:num_iters

    % Perform gradient descent using batch update algorithm. Simultaneously update theta0 and theta 1
    for m_its = 1:m
        sum_theta_zero = sum_theta_zero + ( theta(1) + theta(2)*X(m_its,2) - y(m_its));
    end
    
    for m_its = 1:m
        sum_theta_one = sum_theta_one + ( theta(1) + theta(2)*X(m_its,2) - y(m_its))*X(m_its,2);
    end
    
       
     temp1 = theta(1) - alpha*(1/m)*sum_theta_zero;
     temp2 = theta(2) - alpha*(1/m)*sum_theta_one;
     
     theta(1) = temp1;
     theta(2) = temp2;
     
    J_history(iter) = computeCost(X, y, theta);
    
end

end
