clc;
clear all;

N=50;
Pc=0.9;
Pm=0.005;
ITER=50;
m=2;
BS=[10 10];
L=sum(BS);
Lo=[-4 -1.5];
Hi=[2 1];
Population=round(rand(N,L));
best_so_far=[];
Average_fitness=[];
for it=1:ITER
    [real_val]=chrom_decode(Population,N,L,BS,m,Lo,Hi);
    [selection_probability,fit,ave_fit,max_fit,opt_sol]=fit_eval(real_val,N,m);
    if it==1
        best_so_far(it)=max_fit;
        final_sol=opt_sol;
    elseif max_fit>best_so_far(it-1)
        best_so_far(it)=max_fit;
        final_sol=opt_sol;
    else
        best_so_far(it)=best_so_far(it-1);
    end
    Average_fitness(it)=ave_fit;
    [mating_pool]=g_roulette_wheel(Population,N,selection_probability);
    [new_pop]=g_crossover(mating_pool,Pc,N,L);
    [population]=g_mutation(new_pop,Pm,N,L);
end

disp("Final solution and optimum fitness:");
disp(final_sol);
disp("Best_so_far:");
disp(best_so_far(end));

x=1:ITER;
figure;
plot(x,best_so_far,"k",x,Average_fitness,".-k");
xlabel("Generation");
ylabel("Fitness Function");
legend("Best-so-far","Average fitness");