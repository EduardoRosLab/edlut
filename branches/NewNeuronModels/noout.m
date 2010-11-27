output=load('Register.dat');
lx=max(output(:,1));

spk=sort(output(:,2)');

nspk=[];
if length(spk) > 0
    reps=1;
    nspk(1)=spk(1);
else
    reps=0;
end
for n=2:length(spk),
    if spk(n-1) ~= spk(n)
        reps=reps+1;
        nspk(reps)=spk(n);
    end
end
if reps>18
    display('Too many neurons to be displayed properly, use "nooutbig.m" instead');
end
for n=1:reps
    tspkpot=output(find(output(:,2)==nspk(n)),[1 3]);
    tspk=tspkpot(find(tspkpot(:,2)==1),1);
    subplot(reps,1,n),stem(tspk,-0.081*ones(length(tspk),1),'.');    
    hold on;
    tpot=tspkpot(find(tspkpot(:,2)<0.5),1:2);
    subplot(reps,1,n),plot(tpot(:,1),tpot(:,2),'kx-');
    ylabel(nspk(n));
    axis([0 lx+0.0001 -0.08 0.04]);
end

