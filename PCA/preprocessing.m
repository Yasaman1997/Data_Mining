u1=transpose([10,10])
u2=transpose([22,10])
sigma=[4,4;4,9]

%rng default  % For reproducibility

r1= mvnrnd(u1,sigma,1000)
r2 = mvnrnd(u2,sigma,1000);

f1=figure;
plot(r1(:,:),r1(:,:),'*')
hold
plot(r1(:,0),r1(:,1),'+')


 

hold


plot(r2(:,:),r2(:,:),'+')
hold
plot(r2(:,0),r2(:,1),'+')
%hold
%plot(r2(:,2),r2(:,3),'-o')




