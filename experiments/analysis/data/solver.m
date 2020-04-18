%requires yalmip and spdt3 installation to run solver

%example:
% A = [-1 2 0;-3 -4 1;0 0 -2];
% P = sdpvar(3,3);
% F = [P >= 0, A'*P+P*A <= 0];
% F = [F, trace(P) == 1];
% optimize(F);
% Pfeasible = value(P);
% check(F)

%test:
x_obs = cov(train')
x_test = cov(test')
dimx = size(x_obs,1)
S = sdpvar(dimx,dimx)
L = sdpvar(dimx,dimx)

Constr = [S-L >= 1, L >= 0]

lambda = 0.001
gamma = 0.001

Obj = -logdet(S-L) + trace(x_obs*(S-L)) + lambda * (gamma*norm(S,1)+trace(L))
optimize(Constr,Obj);
s = value(S)
l = value(L)
value(Obj)

objtrain = -log(det(s-l)) + trace(x_obs*(s-l)) + lambda * (gamma*norm(s,1)+trace(l))
objtest = -log(det(s-l)) + trace(x_test*(s-l)) + lambda * (gamma*norm(s,1)+trace(l))

%%%%%%%%%%%%
%param sweep:
x_obs = cov(train')
x_test = cov(test')
dimx = size(x_obs,1)
count1 = 0
count2 = 0
sweep_start_lambda = 0.001
sweep_step_lambda = 0.05
sweep_end_lambda = 1.0

sweep_start_gamma = 0.001
sweep_step_gamma = 0.05
sweep_end_gamma = 1.0

for lambda=sweep_start_lambda:sweep_step_lambda:sweep_end_lambda
    count1 = count1 + 1
end

for gamma=sweep_start_gamma:sweep_step_gamma:sweep_end_gamma
    count2 = count2 + 1
end

heatmap_test = zeros(count1,count2);
% heatmap_train = zeros(count1,count2);
heatmap_lambda = zeros(count1,count2);
heatmap_gamma = zeros(count1,count2);
idx_lambda = 1;
obj_best = 99999
lambda_best = 99999
gamma_best = 99999

%(idx_lambda,idx_gamma) -> matrix
s_map = zeros(count1,count2,dimx,dimx);
l_map = zeros(count1,count2,dimx,dimx);

for lambda=sweep_start_lambda:sweep_step_lambda:sweep_end_lambda
    idx_gamma = 1;
    for gamma=sweep_start_gamma:sweep_step_gamma:sweep_end_gamma
         idx_lambda
         idx_gamma
        x_obs = cov(train');
        x_test = cov(test');
        dimx = size(x_obs,1);
        S = sdpvar(dimx,dimx);
        L = sdpvar(dimx,dimx);

        Constr = [S-L >= 1, L >= 0]
        Obj = -logdet(S-L) + trace(x_obs*(S-L)) + lambda * (gamma*norm(S,1)+trace(L))
        
        optimize(Constr,Obj);
        s = value(S);
        l = value(L);
        xvalid = -log(det(s-l)) + trace(x_test*(s-l)) + lambda * (gamma*norm(s,1)+trace(l))

        if (xvalid < 0)
            heatmap_test(idx_lambda,idx_gamma) = xvalid;
            s_map(idx_lambda,idx_gamma,:,:) = s;
            l_map(idx_lambda,idx_gamma,:,:) = l;
        end
        
        heatmap_lambda(idx_lambda,idx_gamma) = lambda;
        heatmap_gamma(idx_lambda,idx_gamma) = gamma;
        if xvalid < obj_best
           obj_best = xvalid;
           sbest = s;
           lbest = l;
           lambda_best = lambda;
           gamma_best = gamma;
        end
        idx_gamma = idx_gamma + 1;
    end
    idx_lambda = idx_lambda + 1;
end

s=sbest
l=lbest

save solve_tiny_2014_2018_weekly.mat obj_best s l lambda_best gamma_best heatmap_test heatmap_gamma heatmap_lambda s_map l_map

imagesc(heatmap_test);
imagesc(s)
imagesc(l)

%peek at a l matrix
squeeze(l_map(1,1,:,:))


% Obj = -logdet(S-L) + trace(x_obs*(S-L)) + lambda * (gamma*norm(S,1)+trace(L))
% 
% optimize(Constr,Obj)
% s = value(S)
% l = value(L)
% value(Obj)
% eigs(l)
% eigs(s-l)
% rank(s)
% xvalid = -log(det(s-l)) + trace(x_test*(s-l)) + lambda * (gamma*norm(s,1)+trace(l))
% 
% save solved.mat s l