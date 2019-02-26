function[] = MC_BIC_sim_func(n,mc_num,fig,f,h,h0,lambdamax,gridpts,gs)
    mc_num = round(str2double(mc_num));
    n = round(str2double(n));
    f = round(str2double(f));
    h = str2num(h);
    h0 = str2num(h0);
    lambdamax = str2double(lambdamax);
    gridpts = round(str2double(gridpts));
    gs = str2double(gs);
    int_err = zeros(mc_num,1);% prediction err for each MC; our method
    int_err0 = zeros(mc_num,1);%prediction err for each MC; naive NH;
    int_tr_err = zeros(mc_num,1);%mse for each MC;our method;
    int_tr_err0 = zeros(mc_num,1);%mse for each MC;naive NH;
    r = zeros(mc_num,1);%rank
    
    if strcmp(fig,'triangle')
       triangleprelim3=imread('device4-20.gif');
       B0 =double(imresize(triangleprelim3, 0.114));
    elseif strcmp(fig,'butterfly')
        shape = imread('butterfly-4.gif');
        B0 = double(imresize(shape, 0.0906));
    else
        shape = imread('cross.gif');
        B0 = double(imresize(shape, 0.114));
    end

    B0 = B0*5;
    var = 1;
    d = size(B0,2);
    x = linspace(0,1,n);
    x = x';
        
    if f==1
        f_x = @(x) x^2*B0;
    else
        f_x = @(x) sinweight(x,d).*B0;
    end
    fx = zeros(d,d,n);
    for i =1:n
        fx(:,:,i) = f_x(x(i));
    end
    %true rank;
    r0 = rank(fx(:,:,n/2));
    parfor t = 1:mc_num
        s = RandStream('mt19937ar','Seed',t);
        RandStream.setGlobalStream(s);
        
        err = zeros(n,1);%prediction error of our method
        tr_err = zeros(n,1);%mse of our method
        err0 = zeros(n,1);%prediction error of naive NH
        tr_err0 = zeros(n,1);%mse of naive NH
          
        Y = fx + normrnd(0,1,[d,d,n])*sqrt(var);%train
        Ynew = fx + normrnd(0,1,[d,d,n])*sqrt(var);%test
        
        BIC = zeros(length(h),gridpts);
        BIC0 = zeros(length(h0),1);
        
        for i = 1:length(h)
            for j = 1:gridpts
                lambda  = lambdamax*gs^(j-1);
                [~,BIC(i,j),~,~]  = BIC_select_func(x,Y,lambda,h(i));
            end
        end
        
        for i = 1:length(h0)
            [~,BIC0(i),~,~] = naive_BIC_func(x,Y,h0(i));
        end
       

        [row,col] = find(BIC == min(min(BIC)));
        h_min = h(row);
        lambda = lambdamax*gs^(col-1);

        h0_min = h0(BIC0==min(BIC0));
        hvec = ones(n,1)*h_min;
        hvec0 = ones(n,1)*h0_min;

        %initialized rank
        r_tmp = 0;

        for i = 1:n
            kerv = arrayfun(@kern,x(i)-x,hvec);
            kerv0 = arrayfun(@kern,x(i)-x,hvec0);

            ker_sum = sum(kerv);
            ker_sum0 = sum(kerv0);
            lambda_temp = lambda/(2*ker_sum);
            wt_y = zeros(d,d);%weighted Y
            wt_y0 = wt_y;
            for j = 1:n
                wt_y = wt_y + kerv(j)*Y(:,:,j);
                wt_y0 = wt_y0 + kerv0(j)*Y(:,:,j);
            end
            wt_y = wt_y/ker_sum;
            wt_y0 = wt_y0/ker_sum0;
            [U,S,V] = svd(wt_y);
            diag_S = diag(S);
            diag_S_shred = subplus(diag_S - lambda_temp);%positive part

            DS = diag(diag_S_shred);
            yhat = U*DS*V';
            yhat0 = wt_y0;%estimate from naive NH estimator

            err(i) = norm(Ynew(:,:,i) - yhat,'fro')^2; 
            tr_err(i) = norm(Y(:,:,i) - yhat,'fro')^2; 
            err0(i) = norm(Ynew(:,:,i) - yhat0,'fro')^2;
            tr_err0(i) = norm(Y(:,:,i) - yhat0,'fro')^2;

            %selected rank
            r_tmp = r_tmp + sum(diag_S_shred>0);
        end

        int_err(t) = mean(err);%prediction err of our method
        int_err0(t) = mean(err0);% prediction err of naive estimate
        int_tr_err(t) = mean(tr_err);%mse of our method
        int_tr_err0(t) = mean(tr_err0);% mse of naive estimate

        %selected rank
        r(t) = r_tmp/n;
    
    end
    %prediction err
    %our method
    mean_int_err = mean(int_err);
    se_err = std(int_err)/sqrt(mc_num);
    %naive NW
    mean_int_err0 = mean(int_err0);
    se_err0 = std(int_err0)/sqrt(mc_num);

    %mse
    %our method
    mean_int_tr_err = mean(int_tr_err);
    se_tr_err = std(int_tr_err)/sqrt(mc_num);
    %naive NW
    mean_int_tr_err0 = mean(int_tr_err0);
    se_tr_err0 = std(int_tr_err0)/sqrt(mc_num);
    
    %rank
    mean_r = mean(r);
    se_r = std(r)/sqrt(mc_num);
    
    %true rank
    fprintf('Our method:\n');
    fprintf('%.4f\n', [mean_int_err,se_err]);
    fprintf('\n');
    fprintf('%.4f\n', [mean_int_tr_err,se_tr_err]);
    fprintf('\n');
    fprintf('%.4f\n', [mean_r,se_r,r0]);
    fprintf('\n');
    fprintf('NW method:\n');
    fprintf('%.4f\n', [mean_int_err0,se_err0]);
    fprintf('\n');
    fprintf('%.4f\n',[mean_int_tr_err0,se_tr_err0]);
end