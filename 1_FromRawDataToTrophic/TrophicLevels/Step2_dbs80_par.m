
atlas='dbs80';

load(['fdiff_' group '_' atlas '.mat']);
load([group '_' atlas '.mat']);

parpool('local', 6);

epsFC=0.0004;
epsFCtau=0.0001;
maxC=0.2;
load([group '_connectome.mat']);
C=SC;
C = C/max(max(C))*maxC;

NSUB=size(data,2);

N=size(data{1},1);

indexN=1:N;
TR=3.0;
Tau=1.0;
sigma=0.01;

Isubdiag = find(tril(ones(N),-1));

% Bandpass filter settings
fnq=1/(2*TR);                 % Nyquist frequency
flp = 0.008;                    % lowpass frequency of filter (Hz)
fhi = 0.08;                    % highpass
Wn=[flp/fnq fhi/fnq];         % butterworth bandpass non-dimensional frequency
k=2;                          % 2nd order butterworth filter
[bfilt,afilt]=butter(k,Wn);   % construct the filter


%% Group
FC = zeros(NSUB, N, N);
COVtau = zeros(NSUB, N, N);
parfor nsub=1:NSUB

    ts=data{nsub};
    
    for seed=1:N
        ts(seed,:)=detrend(ts(seed,:)-mean(ts(seed,:), 'omitnan'));
    end
    
    ts2=ts(indexN,10:end-10);
    Tm=size(ts2,2);
    FCemp=corrcoef(ts2');
    FC(nsub,:,:)=FCemp;
    COVemp=cov(ts2');
    
    tst=ts2';
    sigratio = zeros(N, N);
    COVtauemp = zeros(N, N);
    
    for i=1:N
        for j=1:N
            sigratio(i,j)=1/sqrt(COVemp(i,i))/sqrt(COVemp(j,j));
            [clag, lags] = xcov(tst(:,i),tst(:,j),Tau);
            indx=find(lags==Tau);
            COVtauemp(i,j)=clag(indx)/size(tst,1);
        end
    end
    
    COVtauemp=COVtauemp.*sigratio;
    COVtau(nsub,:,:)=COVtauemp;
end
FCemp=squeeze(mean(FC));
COVtauemp=squeeze(mean(COVtau));
Cnew=C;
olderror=100000;
errorFC=zeros(1,5000);
errorCOVtau=zeros(1,5000);
for iter=1:5000
    
    % Linear Hopf FC
    [FCsim,COVsim,COVsimtotal,A]=hopf_int(Cnew,f_diff,sigma);
    COVtausim=expm((Tau*TR)*A)*COVsimtotal;
    COVtausim=COVtausim(1:N,1:N);
    
    sigratiosim = zeros(N, N);
    for i=1:N
        for j=1:N
            sigratiosim(i,j)=1/sqrt(COVsim(i,i))/sqrt(COVsim(j,j));
        end
    end
    COVtausim=COVtausim.*sigratiosim;
    errorFC(iter)=mean(mean((FCemp-FCsim).^2));
    errorCOVtau(iter)=mean(mean((COVtauemp-COVtausim).^2));

    if mod(iter,100)<0.1
        errornow=mean(mean((FCemp-FCsim).^2))+mean(mean((COVtauemp-COVtausim).^2));
        if  (olderror-errornow)/errornow<0.001
            break;
        end
        if  olderror<errornow
            break;
        end
        olderror=errornow;
    end

    for i=1:N  %% learning
        for j=1:N
            if (C(i,j)>0 || (strcmp(atlas, 'dbs80') && j == N - i + 1))
                    Cnew(i,j)=Cnew(i,j)+epsFC*(FCemp(i,j)-FCsim(i,j)) ...
                        +epsFCtau*(COVtauemp(i,j)-COVtausim(i,j));
                if Cnew(i,j)<0
                    Cnew(i,j)=0;
                end
            end
        end
    end
    Cnew = Cnew/max(max(Cnew))*maxC;
end
Ceffgroup=Cnew;

%% Individual GEC
Ceff = zeros(NSUB, N, N);
fittFC = zeros(1, NSUB);
fittCVtau = zeros(1, NSUB);
parfor nsub=1:NSUB
    disp(['GEC subject: ' num2str(nsub)]);

    ts=data{nsub};
    
    for seed=1:N
        ts(seed,:)=detrend(ts(seed,:)-mean(ts(seed,:), 'omitnan'));
    end
    
    ts2=ts(indexN,10:end-10);
    Tm=size(ts2,2);
    FCemp=corrcoef(ts2');
    FC(nsub,:,:)=FCemp;
    COVemp=cov(ts2');
    
    tst=ts2';
    sigratio = zeros(N, N);
    COVtauemp = zeros(N, N);
    
    for i=1:N
        for j=1:N
            sigratio(i,j)=1/sqrt(COVemp(i,i))/sqrt(COVemp(j,j));
            [clag, lags] = xcov(tst(:,i),tst(:,j),Tau);
            indx=find(lags==Tau);
            COVtauemp(i,j)=clag(indx)/size(tst,1);
        end
    end
    COVtauemp=COVtauemp.*sigratio;
    
    Cnew=Ceffgroup;
    olderror=100000;
    sigratiosim = zeros(N, N);
    for iter=1:5000
        % Linear Hopf FC
        [FCsim,COVsim,COVsimtotal,A]=hopf_int(Cnew,f_diff,sigma);
        COVtausim=expm((Tau*TR)*A)*COVsimtotal;
        COVtausim=COVtausim(1:N,1:N);
        
        for i=1:N
            for j=1:N
                sigratiosim(i,j)=1/sqrt(COVsim(i,i))/sqrt(COVsim(j,j));
            end
        end
        COVtausim=COVtausim.*sigratiosim;

        if mod(iter,100)<0.1
            errornow=mean(mean((FCemp-FCsim).^2))+mean(mean((COVtauemp-COVtausim).^2));
            if  (olderror-errornow)/errornow<0.001
                break;
            end
            if  olderror<errornow
                break;
            end
            olderror=errornow;
        end

        for i=1:N  %% learning
            for j=1:N
                if (C(i,j)>0 || (strcmp(atlas, 'dbs80') && j == N - i + 1))
                        
                        Cnew(i,j)=Cnew(i,j)+epsFC*(FCemp(i,j)-FCsim(i,j)) ...
                            +epsFCtau*(COVtauemp(i,j)-COVtausim(i,j));
                    if Cnew(i,j)<0
                        Cnew(i,j)=0;
                    end
                end
            end
        end
        Cnew = Cnew/max(max(Cnew))*maxC;
    end
    Ceff=Cnew;
    Ceff(nsub,:,:)=Ceff;
    [FCsim,COVsim,COVsimtotal,A]=hopf_int(Ceff,f_diff,sigma);
    fittFC(nsub)=corr2(FCemp(Isubdiag),FCsim(Isubdiag));
    COVtausim=expm((Tau*TR)*A)*COVsimtotal;
    COVtausim=COVtausim(1:N,1:N);
    for i=1:N
        for j=1:N
            sigratiosim(i,j)=1/sqrt(COVsim(i,i))/sqrt(COVsim(j,j));
        end
    end
    COVtausim=COVtausim.*sigratiosim;
    fittCVtau(nsub)=corr2(COVtauemp(Isubdiag),COVtausim(Isubdiag));
end


%% Trophic Levels

hierarchicallevels = zeros(NSUB, N);
directedness = zeros(1, NSUB);
for nsub=1:NSUB
    Ceff=squeeze(Ceff(nsub,:,:));
    A=Ceff';
    d=sum(A)';
    delta=sum(A,2);
    u=d+delta;
    v=delta-d;
    Lambda=diag(u)-A-A';
    Lambda(1,1)=0;
    gamma=linsolve(Lambda,v);
    hierarchicallevels(nsub,:)=gamma';
    gamma=gamma-min(gamma);
    H = gamma - gamma';
    F0 = sum(sum(A .* (H - 1).^2)) / sum(A(:));
    directedness(nsub)=1-F0;
end

save(['TL_' group '_' atlas '.mat']);
delete(gcp('nocreate'));
