function [df,BIC,err,yhat] = BIC_select_func(x,Y,lambda,h)

[n,~] = size(x);
d = size(Y,1);

err = 0;%our method

%% our method
yhat = zeros(d,d,n);
kerv = zeros(n,n);
%hvec = ones(n,1)*h;
df_tmp = zeros(n,1);
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
    wt_y = wt_y/ker_sum;
    [U,S,V] = svd(wt_y);
    diag_S = diag(S);
    lambda_tmp = lambda/(2*ker_sum);
    diag_S_shred = subplus(diag_S - lambda_tmp);%positive part
    DS = diag(diag_S_shred);
    yhat(:,:,i) = U*DS*V';       
    err = err + norm(Y(:,:,i) - yhat(:,:,i),'fro')^2;
    ind_tmp1 = find(diag_S_shred>0);  

    
    for j = ind_tmp1'
        ind_tmp2 = [1:(j-1),(j+1):d];
        df_tmp(i) = df_tmp(i) + 1+ 2*sum((diag_S(j)*diag_S_shred(j))./(diag_S(j)^2 ...
            - diag_S(ind_tmp2).^2));    
    end 
end

df = sum(df_tmp./sum(kerv,2))*kern(0,h);
BIC = (n*d^2)*log(err/(n*d^2)) + log(n*d^2)*df;
err = err/n;
end