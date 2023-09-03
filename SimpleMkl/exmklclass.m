 clc;
clear all
close all
nbiter=20;
ratio=0.7;
% data='Liver22';
 % data='ionosphere';
data='WPBC';
% data='pima';
% data='Magic';
% data='banknote';
% data='haberman';
% data='Xynew';
% data='Sonar';
load([data ]);
C = [100];
x=X;
verbose=1;

options.algo='svmclass'; % Choice of algorithm in mklsvm can be either
                         % 'svmclass' or 'svmreg'
%------------------------------------------------------
% choosing the stopping criterion
%------------------------------------------------------
options.stopvariation=0; % use variation of weights for stopping criterion 
options.stopKKT=0;       % set to 1 if you use KKTcondition for stopping criterion    
options.stopdualitygap=1; % set to 1 for using duality gap for stopping criterion

%------------------------------------------------------
% choosing the stopping criterion value
%------------------------------------------------------
options.seuildiffsigma=1e-2;        % stopping criterion for weight variation 
options.seuildiffconstraint=0.1;    % stopping criterion for KKT
options.seuildualitygap=0.01;       % stopping criterion for duality gap

%------------------------------------------------------
% Setting some numerical parameters 
%------------------------------------------------------
options.goldensearch_deltmax=1e-1; % initial precision of golden section search
options.numericalprecision=1e-8;   % numerical precision weights below this value
                                   % are set to zero 
options.lambdareg = 1e-8;          % ridge added to kernel matrix 

%------------------------------------------------------
% some algorithms paramaters
%------------------------------------------------------
options.firstbasevariable='first'; % tie breaking method for choosing the base 
                                   % variable in the reduced gradient method 
options.nbitermax=500;             % maximal number of iteration  
options.seuil=0;                   % forcing to zero weights lower than this 
options.seuilitermax=10;           % value, for iterations lower than this one 

options.miniter=0;                 % minimal number of iterations 
options.verbosesvm=0;              % verbosity of inner svm algorithm 

%
% Note: set 1 would raise the `strrep`
%       error in vectorize.dll
%       and this error is not able to fix
%       because of the missing .h libraay files
% Modify: MaxisKao @ Sep. 4 2014
options.efficientkernel=0;         % use efficient storage of kernels 


%------------------------------------------------------------------------
%                   Building the kernels parameters
%------------------------------------------------------------------------
% kernelt={'gaussian' 'gaussian' 'poly' 'poly' };
% a=rand(10,1);
% kerneloptionvect={[a] [a]};
% kerneloptionvect={[0.5 1 2 5 7 10 12 15 17 20] [0.5 1 2 5 7 10 12 15 17 20] [1 2 3] [1 2 3]};
% variablevec={'all' 'single' 'all' 'single'};
kernelt={'gaussian' 'gaussian' };
% kerneloptionvect={[0.5 1 2 5 7 10 12 15 17 20] [0.5 1 2 5 7 10 12 15 17 20]};
kerneloptionvect={[0.1 0.2 0.3 0.5 0.7 1 1.2 1.5 1.7 2] [0.1 0.2 0.3 0.5 0.7 1 1.2 1.5 1.7 2]};
% kerneloptionvect={[0.14 0.0995 0.0161 0.0409 0.0156 0.0156 0.1221 0.1175 0.0539 0.1247] [0.14 0.0995 0.0161 0.0409 0.0156 0.0156 0.1221 0.1175 0.0539 0.1247]} ;
variablevec={'all' 'single'};

classcode=[1 -1];

[nbdata,dim]=size(x);
 % dim=1;
nbtrain=floor(nbdata*ratio);
 rand('state',0);

for i=1: nbiter
     
    [xapp,yapp,xtest,ytest,indice]=CreateDataAppTest(x, y, nbtrain,classcode);%划分训练集和测试集，并记录训练和测试的索引
    [xapp,xtest]=normalizemeanstd(xapp,xtest);%对输入数据进行标准化处理，均值设置为0，标准差设置为1.
    [kernel,kerneloptionvec,variableveccell]=CreateKernelListWithVariable(variablevec,dim,kernelt,kerneloptionvect);
     % kernel={'gaussian'};%此为10核情况，L=10
    [Weight,InfoKernel]=UnitTraceNormalization(xapp,kernel,kerneloptionvec,variableveccell);%单位迹归一化
    K=mklkernel(xapp,InfoKernel,Weight,options);

    
    tic
    [beta,w,b,posw,story(i),obj(i)] = mklsvm(K,yapp,C,options,verbose);
    timelasso(i)=toc
 
     Kt=mklkernel(xtest,InfoKernel,Weight,options,xapp(posw,:),beta);

     
   
     ypred=Kt*w+b;
     ac(i)=size(posw,1);
      bc(i)=mean(sign(ypred)==ytest)

      d(i)=nnz(beta);
    
end
acc_junzhi=mean(bc)
acc_std=std(bc)
nsv_junzhi=mean(ac)
nsv_std=std(ac)
time_junzhi=mean(timelasso)
time_std=std(timelasso)
d_junzhi=mean(d)
d_std=std(d)

