function InfoKernel=Unit(kernelvec,kerneloptionvec,variablevec)
nbk=1;
for i=1:length(kernelvec)
    % i
    for k=1:length(kerneloptionvec{i})
            InfoKernel(nbk).kernel=kernelvec{i};
            InfoKernel(nbk).kerneloption=kerneloptionvec{i}(k);
            InfoKernel(nbk).variable=variablevec{i};
            nbk=nbk+1;
    end
end

end