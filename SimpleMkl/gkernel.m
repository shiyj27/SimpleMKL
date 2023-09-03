


A = [1 2 310; 4 5 6; 7 8 0.9];
% B=[1 2 3; 4 5 6; 7 8 9];
B = [101 11 102; 13 14 15; 16 17 180];
sigma = 1;  % you can choose a suitable value for sigma
K = gaussian_kernel(A, B, sigma)
sig=1;
Xt=A;
X1=B;
norms1=sum(Xt'.^2)';
            [m2,~]=size(X1);
            norms2=sum(X1'.^2);
            [mt,~]=size(Xt);   
%             g_l_test=zeros(mt,L);
%             KK=cell(L,1);
%             Dy1=diag(y1);
%             for l=1:L
            KK=exp((-norms1*ones(1,m2)-ones(mt,1)*norms2+2*Xt*(X1'))/(2))
%             end
%             




function K = gaussian_kernel(X1, X2, sigma)
    m = size(X1, 1);
    n = size(X2, 1);
    K = zeros(m, n);
    for i = 1:m
        for j = 1:n
            K(i, j) = exp(-norm(X1(i, :) - X2(j, :))^2 / (2 * sigma^2));
        end
    end
end
