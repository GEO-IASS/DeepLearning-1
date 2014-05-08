function [cost,grad,features] = sparseAutoencoderLinearCost(theta, visibleSize, hiddenSize, ...
                                                            lambda, sparsityParam, beta, data)
% -------------------- YOUR CODE HERE --------------------
% Instructions:
%   Copy sparseAutoencoderCost in sparseAutoencoderCost.m from your
%   earlier exercise onto this file, renaming the function to
%   sparseAutoencoderLinearCost, and changing the autoencoder to use a
%   linear decoder.
% -------------------- YOUR CODE HERE --------------------                                    

% visibleSize: the number of input units 
% hiddenSize: the number of hidden units 
% lambda: weight decay parameter
% sparsityParam: the desired average activation for the hidden units
% beta: weight of sparsity penalty term
% data: Our NxM matrix containing the training data.  So, data(:,i) is the i-th training example. 
  
% The input theta is a vector (because minFunc expects the parameters to be a vector). 
% We first convert theta to the (W1, W2, b1, b2) matrix/vector format, so that this 
% follows the notation convention of the lecture notes. 

W1 = reshape(theta(1:hiddenSize*visibleSize), hiddenSize, visibleSize);
W2 = reshape(theta(hiddenSize*visibleSize+1:2*hiddenSize*visibleSize), visibleSize, hiddenSize);
b1 = theta(2*hiddenSize*visibleSize+1:2*hiddenSize*visibleSize+hiddenSize);
b2 = theta(2*hiddenSize*visibleSize+hiddenSize+1:end);

% Cost and gradient variables (your code needs to compute these values). 
% Here, we initialize them to zeros. 
cost = 0;
W1grad = zeros(size(W1)); 
W2grad = zeros(size(W2));
b1grad = zeros(size(b1)); 
b2grad = zeros(size(b2));

% Store the number of samples in 'm'
m = size(data,2);

% Computing activations (feedforward)
z2 = W1 * data + repmat(b1,1,m);
a2 = sigmoid(z2);
z3 = W2 * a2 + repmat(b2,1,m);
a3 = z3;	% linear acivation 

% Store in 'pj' the average activation values of hidden units
pj = sum(a2,2) ./ m;

% Computing errors (backpropagation)  
error3 = -(data-a3);
error2 = ((W2'*error3) + repmat(beta.*(-(sparsityParam./pj)+((1-sparsityParam)./(1-pj))),1,m)) .* (a2.*(1-a2));

% Computing gradients
W2grad = (error3*(a2') ./ m) + (lambda * W2);
b2grad = sum(error3,2) ./ m;
W1grad = (error2*(data') ./ m) + (lambda * W1);
b1grad = sum(error2,2) ./ m;

% Computing the cost + weightdecay + kldivergence
weightdecay = sum(sum(W1.^2)) + sum(sum(W2.^2));
kldivergence = sum((sparsityParam.*log(sparsityParam./pj)) + ((1-sparsityParam).*log((1-sparsityParam)./(1-pj))));
cost = (sum(sum(((a3-data).^2)./2)) / m) + (lambda/2 * weightdecay) + (beta * kldivergence);

%-------------------------------------------------------------------
% After computing the cost and gradient, we will convert the gradients back
% to a vector format (suitable for minFunc).  Specifically, we will unroll
% your gradient matrices into a vector.

grad = [W1grad(:) ; W2grad(:) ; b1grad(:) ; b2grad(:)];

end

%-------------------------------------------------------------------
% Here's an implementation of the sigmoid function, which you may find useful
% in your computation of the costs and the gradients.  This inputs a (row or
% column) vector (say (z1, z2, z3)) and returns (f(z1), f(z2), f(z3)). 

function sigm = sigmoid(x)
  
    sigm = 1 ./ (1 + exp(-x));
end