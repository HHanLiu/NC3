% function [B,Lambda,W,D,L,L_sym,DIST] = AffinityMatrix(X,anchor,k0)
function [B,Lambda,W,DIST,time] = AffinityMatrix(X,anchor,k0)

%Input: X: data matrix, n ¡Á d
%       anchor: anchor matrix£¬m ¡Á d
%       k0: k-NN 
%Output: B: anchor-sample graph
%        Lambda: sum of column of B
%        W: affinity matrix
%        D: degree matrix
%        L,L_sym: Laplacian matrix, normalized Laplacian matrix

a = tic;
n = size(X,1);% data 
m = size(anchor,1);% anchor 
B = zeros(n,m);
dist = L2_distance_1(X',anchor');
DIST = dist;
dist = dist.^2;

% % Clustering and projected clustering with adaptive neighbors
for i = 1:n
    if(ismember(X(i,:),anchor,'rows'))
        [~,ind]=min(dist(i,:));
        B(i,:)=zeros(1,m);
        B(i,ind)=1;
    else
        [val,ind] = mink(dist(i,:),k0);
        val2 = mink(dist(i,:),k0+1);
        kt1 = max(val2);
        val = kt1 - val;
        val = val./sum(val);
        t = zeros(1,m);
        t(ind) = val;
        B(i,:) = t;
    end
end
Lambda = diag(sum(B,1));

W=0;
% W = [zeros(n,n),B;
%     B',zeros(m,m)];
% D = [eye(n,n),zeros(n,m)
%     zeros(m,n),Lambda];
% 
% L = D-W;
% L_sym = eye(n+m) - (D^(-1/2) * W * D^(-1/2));
time = toc(a);

end

