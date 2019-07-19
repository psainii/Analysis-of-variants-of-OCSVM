function ker = RBF(x,xj,gamma)
[n,m] = size(x);
[a,b] = size(xj);
ker = zeros(a,n);
for i = 1:n
    for j = 1:a
        ker(j,i) = norm(x(i,:) - xj(j,:));
    end
end
ker = exp(ker.^2*(-gamma));
end
