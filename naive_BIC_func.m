function [df,BIC,err,yhat] = naive_BIC_func(x,Y,h)

[n,~] = size(x);
d = size(Y,1);

err = 0;%our method


%% our method
yhat = zeros(d,d,n);
kerv = zeros(n,n);
hvec = ones(n,1)*h;
% for i = 1:n
%     kerv(i,:)  = arrayfun(@kern,x(i)-x,hvec);%line i: x(i) 
% end

for i = 1:n
    for j = 1:n
        kerv(i,j)  = kern(x(i,:)-x(j,:),h);%line i: x(i)
    end
end

for i = 1:n
    wt_y =  zeros(d,d);
    for j = 1:n
        wt_y = wt_y + kerv(i,j)*Y(:,:,j);
    end
    ker_sum = sum(kerv(i,:));
    yhat(:,:,i) = wt_y/ker_sum; 
    err = err+ norm(Y(:,:,i) - yhat(:,:,i),'fro')^2;
end

df = d^2*sum(1./sum(kerv,2))*kern(0,h);
BIC = (n*d^2)*log(err/(n*d^2)) + log(n*d^2)*df;
err = mean(err);
end