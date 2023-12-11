function Y = reflattenHack(X)
S = size(X,1);
C = size(X,2);
B = size(X,3);
Y = reshape(X,sqrt(S),sqrt(S),C,B);
Y = permute(Y,[2 1 3 4]);
Y = dlarray(reshape(Y, S, C, B));
end
