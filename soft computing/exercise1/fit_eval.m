function[selection_probability,fit,ave_fit,max_fit,opt_sol]=fit_eval(real_val,N,m)
    for i=1:N
        x(1)=real_val(i,1);
        x(2)=real_val(i,2);
        fit(i)=(1+cos(2*pi*x(1)*x(2)))*exp(-(abs(x(1))+abs(x(2)))/2);
    end
    selection_probability=fit/sum(fit);
    ave_fit=mean(fit);
    [max_fit,max_loc]=max(fit);
    opt_sol=real_val(max_loc,:);
return;