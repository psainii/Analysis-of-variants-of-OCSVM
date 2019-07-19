function [alpha,rho,Q] = OCSVM(x,nu,gamma)
[n,m] = size(x);
Q = RBF(x,x,gamma); % Kernel Matrix using RBF function.
alpha = ones(n,1); % Lagrange multipliers.
Aeq = ones(1,n);
beq = 1;
lb = zeros(n,1);
ub = ones(n,1)/(n*nu);
options = optimoptions('quadprog','Algorithm','interior-point-convex','Display','off');
alpha = quadprog(Q,[],[],[],Aeq,beq,lb,ub,[],options); % Solving dual problem using quadprog.
rho = 0; 
nos = 0;
for i = 1:n
    if alpha(i)==0 || alpha(i)<1e-5 || alpha(i)==1/(n*nu) % If the value of alpha is zero,1/n*mu or less than 1e-5 then make it zero.
        alpha(i)=0;
    else
        rho = rho + Q(i,:)*alpha; % Add the rho corresponding to the support vectors.
        nos = nos + 1;
    end
end
rho = rho / nos; % Take the mean of rho values.
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%How to use the function for One-class Support Vector Machine%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Input x is the training feature matrix of n examples and m dimension.%%
%% Input nu sets the upper bound on number of outliers.                 %%
%% Input gamma is the parameter corresponding to the RBF kernel.        %%
%% gamma = 1/(2*(sigma^2)), where sigma is the width parameter.         %%
%% Output alpha are the lagrangian multipliers. Its dimensions are [n,1]%%
%% Output rho is the lower bound on the number of inliers.              %%
%% Output Q is the Kernel matrix of dimension [n,n].                    %%
%% Decision function can be calculated by Ftrain = sign(Q*alpha - rho)	%%
%% To calculate decision function on testing data,                      %%
%% use Kt = RBF(Xtrain,Xtest,gamma),                                    %%
%% and then Ftest = sign(Kt*alpha-rho)                                  %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%