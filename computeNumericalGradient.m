function numgrad = computeNumericalGradient(J, theta)
% numgrad = computeNumericalGradient(J, theta)
% theta: a vector of parameters
% J: a function that outputs a real-number. Calling y = J(theta) will return the
% function value at theta. 
  
% Initialize numgrad with zeros
grad = zeros(size(theta));
EPSILON = 0.001;
[m n] = size(theta);

%% ---------- YOUR CODE HERE --------------------------------------
% Instructions: 
% Implement numerical gradient checking, and return the result in numgrad.  
% (See Section 2.3 of the lecture notes.)
% You should write code so that numgrad(i) is (the numerical approximation to) the 
% partial derivative of J with respect to the i-th input argument, evaluated at theta.  
% I.e., numgrad(i) should be the (approximately) the partial derivative of J with 
% respect to theta(i).
%                
% Hint: You will probably want to compute the elements of numgrad one at a time. 

for i = 1:m
    eps = zeros(m,1);
    eps(i,1) = 1;
    eps = EPSILON*eps;
    grad(i) = (J(theta + eps) - J(theta - eps))/(2*EPSILON) ;
end

numgrad = grad;

%% ---------------------------------------------------------------


end
