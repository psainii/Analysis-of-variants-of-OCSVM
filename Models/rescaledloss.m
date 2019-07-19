function [rho , alpha] = rescaledloss(S)

% for i=1:8
%     S(:,i)=(S(:,i)-min(S(:,i)))/(max(S(:,i))-min(S(:,i)));
% end
% x_pos = S(S(:,end)==1,1:end-1);
% x_neg = S(S(:,end)==0,1:end-1);
% pos = length(x_pos);
% neg = length(x_neg);
% X_train=[x_pos(1:round(pos*0.2),:);x_neg(1:round(neg*0.9),:)];
% Y_test=[ones(pos-round(pos*0.2),1);zeros(neg-round(neg*0.9),1)];
% Y_train=[ones(round(pos*0.2),1);zeros(round(neg*0.9),1)];
% X_test=[x_pos((round(pos*0.2)+1):end,:);x_neg((round(neg*0.9)+1):end,:)];
x_train=S(:,1:end-1);
y_train=S(:,end);
xpos_train=x_train(y_train==1,:);
ypos_train=y_train(y_train==1,:);
X_train=xpos_train(1:round(0.8*sum(y_train==1)),:);
Y_train=ypos_train(1:round(0.8*sum(y_train==1)),:);
xneg=x_train(y_train==0,:);
yneg=y_train(y_train==0,:);
yneg=-1*(yneg==0);
X_train=[X_train;xneg(1:round(0.05*sum(y_train==0)),:)];
Y_train=[Y_train;yneg(1:round(0.05*sum(y_train==0)),:)];
X_test=[xneg(round(0.05*sum(y_train==0))+1:end,:);xpos_train(round(0.8*sum(y_train==1))+1:end,:)];
Y_test=[yneg(round(0.05*sum(y_train==0))+1:end,:);ypos_train(round(0.8*sum(y_train==1))+1:end,:)];
[n,m] = size(X_train);

% min-max scaler
for i=1:m
    X_train(:,i)=(X_train(:,i)-min(X_train(:,i)))/(max(X_train(:,i))-min(X_train(:,i)));
end

x=X_train;
ita=1;
sigma=20;
nu=0.15;
[n,m] = size(x);
Q = zeros(n,n);
for i=1:n
    for j=1:n
        Q(i,j)=exp(-0.5*norm(x(i,:)-x(j,:))^2/sigma^2);
    end
end
beta=1/(1-exp(-ita));
Q = Q+(1e-10)*eye(n);
e = ones(n,1);
u=-e;
Aeq=e';
Beq=1;
delta=beta*ita*(-u);
iter = 20;
for i = 1:iter
    lb=zeros(n,1);
    ub=(1/(nu*n))*(e.*delta);
    options = optimoptions('quadprog','Algorithm','interior-point-convex','Display','none');
    alpha = quadprog(Q,zeros(n,1),[],[],Aeq ,Beq,lb, ub,[],options);
    ind_sv=find(alpha>1e-5);
    SV=x(ind_sv,:);
    alpha_sv=alpha(ind_sv);
    n_sv=length(alpha_sv);
    K_SV=zeros(n,n_sv);
    for j=1:n
        for k=1:n_sv
            K_SV(j,k)=exp(-0.5*norm(x(j,:)-SV(k,:))^2/sigma^2);
        end
    end
    rho = mean(alpha'*K_SV);
    hinge=max(0,rho-Q*alpha);
    u=-exp(-ita*hinge);
    delta=beta*ita*(-u);
    delta=max(0,delta);
end
end