function [rho alpha] = ramploss(S)


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

%min-max scaler
for i=1:m
    X_train(:,i)=(X_train(:,i)-min(X_train(:,i)))/(max(X_train(:,i))-min(X_train(:,i)));
end

x=X_train;
sigma=0.1;
R_outlier=0.01;
[n,m] = size(x);
for j=1:n
    for k=1:n
        Q(j,k)=exp(-0.5*norm(x(j,:)-x(k,:))^2/sigma^2);
    end
end
r=0.95;
nu=3*R_outlier;
tolerance=0.0001;
alpha = ones(n,1);
Aeq = ones(1,n);
beq = 1;
lb = zeros(n,1);
ub = ones(n,1)/(n*nu);
options = optimoptions('quadprog','Algorithm','interior-point-convex','Display','off');
alpha = quadprog(Q,[],[],[],Aeq,beq,lb,ub,[],options);
tau = zeros(n,1);
Q = Q+1e-10*eye(n);
e = ones(n,1);
Beq = 1+(r-1)*sum(tau);
iter=30;
for i = 1:iter
    lb=-tau;
    ub=(1/(nu*n))*e-tau;
    options = optimoptions('quadprog','Algorithm','interior-point-convex','Display','none');
    alpha = quadprog(Q,zeros(n,1),[],[],Aeq ,Beq,lb,ub,[],options);
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
     rho = mean(K_SV*alpha_sv);
     hinge=max(0,r*rho-Q*alpha);
     tau_new=1/(nu*n)*hinge;
     if(norm(tau_new-tau)<tolerance)
           break;
     end
     tau = tau_new;
end
rho = mean(alpha'*K_SV);
fr = Q*alpha-rho;
[B,I] = sort(fr);
k = ceil(n*R_outlier);
X_in = x(I(k+1:n),:);
X_out=x(I(1:k),:);
[n_new,m] = size(X_in);
Q_in = zeros(n_new,n_new);
for i=1:n_new
     for j=1:n_new
           Q_in(i,j)=exp(-0.5*norm(X_in(i,:)-X_in(j,:))^2/sigma^2);
     end
end
nu = 1/n;

Q_in = Q_in+1e-10*eye(n_new);
e = ones(n_new,1);
lb=zeros(n_new,1);
ub=(1/(nu*n_new))*e;
Aeq=e';
Beq=1;
options = optimoptions('quadprog','Algorithm','interior-point-convex','Display','none');
alpha =  quadprog(Q_in, zeros(n_new,1), [], [], Aeq, Beq, lb,ub, [], options); 
ind_sv=find(alpha>1e-5);
SV=X_in(ind_sv,:);
alpha_sv=alpha(ind_sv);
n_sv=length(alpha_sv);
K_SV1=zeros(n_new,n_sv);
for i=1:n_new
      for j=1:n_sv
            K_SV1(i,j)=exp(-0.5*norm(X_in(i,:)-SV(j,:))^2/sigma^2);
      end
end
rho = mean(K_SV1*alpha_sv);
disp(rho);
disp(alpha);

end