using OrdinaryDiffEq
using ModelingToolkit
using DataDrivenDiffEq
using LinearAlgebra, DiffEqSensitivity, Optim, Statistics
using DiffEqFlux, Flux
using Printf,PGFPlotsX,LaTeXStrings, JLD2
using MAT
include("Numerical_Cont.jl")
#@load "/Users/kyoung/OneDrive - University of Bristol/Documents/Simulations/Flutter_noise/flutter.jld"
# Numerical continuation of the experimental model

## Save data
nh=30
l=6000
vars = matread("CBC_stable_v14_9.mat")
uu=get(vars,"data",1)
ind1=1;ind2=4;
uu1=uu[ind1,1:l]
uu2=uu[ind2,1:l]
mu1=mean(uu[ind1,1:l])
mu2=mean(uu[ind2,1:l])
uu1=uu[ind1,1:l]-mu1*ones(l)
uu2=uu[ind2,1:l]-mu2*ones(l)
t_series=[[transpose(uu1);transpose(uu2)]]
rr=sqrt.(uu1.^2+uu2.^2)
tt=atan.(uu2,uu1)
c=LS_harmonics(rr,tt,1,nh).coeff
AA=c

vars = matread("CBC_stable_v15_6.mat")
uu=get(vars,"data",1)
uu1=uu[ind1,1:l]
uu2=uu[ind2,1:l]
uu1=uu[ind1,1:l]-mu1*ones(l)
uu2=uu[ind2,1:l]-mu2*ones(l)
t_series=vcat(t_series,[[transpose(uu1);transpose(uu2)]])
rr=sqrt.(uu1.^2+uu2.^2)
tt=atan.(uu2,uu1)
c=LS_harmonics(rr,tt,1,nh).coeff
AA=hcat(AA,c)

vars = matread("CBC_stable_v16_5.mat")
uu=get(vars,"data",1)
uu1=uu[ind1,1:l]
uu2=uu[ind2,1:l]
uu1=uu[ind1,1:l]-mu1*ones(l)
uu2=uu[ind2,1:l]-mu2*ones(l)
t_series=vcat(t_series,[[transpose(uu1);transpose(uu2)]])
rr=sqrt.(uu1.^2+uu2.^2)
tt=atan.(uu2,uu1)
c=LS_harmonics(rr,tt,1,nh).coeff
AA=hcat(AA,c)

vars = matread("CBC_stable_v17_3.mat")
uu=get(vars,"data",1)
uu1=uu[ind1,1:l]
uu2=uu[ind2,1:l]
uu1=uu[ind1,1:l]-mu1*ones(l)
uu2=uu[ind2,1:l]-mu2*ones(l)
t_series=vcat(t_series,[[transpose(uu1);transpose(uu2)]])
rr=sqrt.(uu1.^2+uu2.^2)
tt=atan.(uu2,uu1)
c=LS_harmonics(rr,tt,1,nh).coeff
AA=hcat(AA,c)

vars = matread("CBC_unstable_v14_9.mat")
uu=get(vars,"data",1)
ind1=1;ind2=4
uu1=uu[ind1,1:l]
uu2=uu[ind2,1:l]
uu1=uu[ind1,1:l]-mu1*ones(l)
uu2=uu[ind2,1:l]-mu2*ones(l)
t_series=vcat(t_series,[[transpose(uu1);transpose(uu2)]])
rr=sqrt.(uu1.^2+uu2.^2)
tt=atan.(uu2,uu1)
c=LS_harmonics(rr,tt,1,nh).coeff
AA=hcat(AA,c)

vars = matread("CBC_unstable_v15_6.mat")
uu=get(vars,"data",1)
uu1=uu[ind1,1:l]
uu2=uu[ind2,1:l]
uu1=uu[ind1,1:l]-mu1*ones(l)
uu2=uu[ind2,1:l]-mu2*ones(l)
t_series=vcat(t_series,[[transpose(uu1);transpose(uu2)]])
rr=sqrt.(uu1.^2+uu2.^2)
tt=atan.(uu2,uu1)
c=LS_harmonics(rr,tt,1,nh).coeff
AA=hcat(AA,c)

vars = matread("CBC_unstable_v16_5.mat")
uu=get(vars,"data",1)
uu1=uu[ind1,1:l]
uu2=uu[ind2,1:l]
uu1=uu[ind1,1:l]-mu1*ones(l)
uu2=uu[ind2,1:l]-mu2*ones(l)
t_series=vcat(t_series,[[transpose(uu1);transpose(uu2)]])
rr=sqrt.(uu1.^2+uu2.^2)
tt=atan.(uu2,uu1)
c=LS_harmonics(rr,tt,1,nh).coeff
AA=hcat(AA,c)
##

vel_l=4
Vel=[14.9,15.6,16.5,17.3]
Vel2=[14.9,15.6,16.5]
θ_l=300
θ=range(0, stop = 2π, length = θ_l)
coθ=cos.(θ)
siθ=sin.(θ)

plot(uu1,uu2)
d=get_sol_polar(c,l,0,0)
plot!(d.u,d.v)
## Normal form

function nf_dis(U₀,s,Vel,Vel2)
    del=Vel-U₀*ones(length(Vel))
    del2=Vel2-U₀*ones(length(Vel2))
    va2=s*ones(length(Vel))
    va2_2=s*ones(length(Vel2))
    s_amp=sqrt.(va2/2+sqrt.(va2.^2+4*del)/2)
    u_amp=sqrt.(va2_2/2-sqrt.(va2_2.^2+4*del2)/2)

    vl=[s_amp[i]*[coθ';siθ'] for i in 1:length(Vel)]
    vl2=[u_amp[i]*[coθ';siθ'] for i in 1:length(Vel2)]
    (v=vl,v2=vl2)
end

function f_coeff(vlT,Vel,u₀,v₀)
    Pr=zeros(2*nh+1,0)
    for k=1:length(Vel)
        z1=vlT[k][1,:]-u₀*ones(θ_l)
        z2=vlT[k][2,:]-v₀*ones(θ_l)
        theta=atan.(z2,z1)
        r=sqrt.(z1.^2+z2.^2)
        tM=Array{Float64}(undef,0,2*nh+1)
        rr=Array{Float64}(undef,θ_l)
        for j in 1:θ_l
            tM1=Array{Float64}(undef,0,nh+1)
            tM2=Array{Float64}(undef,0,nh)
            tM1_=[cos(theta[j]*i) for i in 1:nh]
            tM2_=[sin(theta[j]*i) for i in 1:nh]
            tM1_=vcat(1,tM1_)
            tM1=vcat(tM1,Transpose(tM1_))
            tM2=vcat(tM2,Transpose(tM2_))
            tM_=hcat(tM1,tM2)
            tM=vcat(tM,tM_)
        end
        MM=Transpose(tM)*tM
#        MM=SMatrix{2*nh+1,2*nh+1}(MM)
        rN=Transpose(tM)*r
#        rN=SMatrix{2*nh+1,1}(rN)
        c=inv(MM)*rN
        Pr=hcat(Pr,c)
        Pr
    end
    Pr
end


function predict_lt(θ_t) #predict the linear transformation
    np1=θ_t[end];np2=3.85
#    nf=nf_dis(U₀,3.85,Vel,Vel2)
    nf=nf_dis(np1,np2,Vel,Vel2)
    vl=nf.v;vl2=nf.v2
    p1,p2,p3,p4,p5,p6=θ_t[1:6]
    T=[p1 p3;p2 p4]

    dis=transpose([p5*ones(θ_l) p6*ones(θ_l)])/scale_f_l
    pn=θ_t
    vlT=[dis*norm(vl[i][:,1])^2+(T+reshape(ann_l([norm(vl[i][:,1]),Vel[i]-np1],θ_t[7:end-1]),2,2)/scale_f2)*(vl[i]) for i in 1:length(Vel)]
    vlT2=[dis*norm(vl2[i][:,1])^2+(T+reshape(ann_l([norm(vl2[i][:,1]),Vel2[i]-np1],θ_t[7:end-1]),2,2)/scale_f2)*(vl2[i]) for i in 1:length(Vel2)]

    Pr=f_coeff(vlT,Vel,0,0)
    Pr2=f_coeff(vlT2,Vel2,0,0)
    PP=hcat(Pr,Pr2)
    PP
end

function predict_lt2(θ_t) #predict the linear transformation
    np1=U₀;np2=3.85
#    nf=nf_dis(U₀,3.85,Vel,Vel2)
    nf=nf_dis(np1,np2,Vel,Vel2)
    vl=nf.v;vl2=nf.v2
    p1,p2,p3,p4,p5,p6=θ_t[1:6]
    T=[p1 p3;p2 p4]

    dis=transpose([p5*ones(θ_l) p6*ones(θ_l)])/scale_f_l
    pn=θ_t
    vlT=[dis*norm(vl[i][:,1])^2+(T+reshape(ann_l([norm(vl[i][:,1]),Vel[i]-np1],θ_t[7:end-1]),2,2)/scale_f2)*(vl[i]) for i in 1:length(Vel)]
    vlT2=[dis*norm(vl2[i][:,1])^2+(T+reshape(ann_l([norm(vl2[i][:,1]),Vel2[i]-np1],θ_t[7:end-1]),2,2)/scale_f2)*(vl2[i]) for i in 1:length(Vel2)]

    Pr=f_coeff(vlT,Vel,0,0)
    Pr2=f_coeff(vlT2,Vel2,0,0)
    PP=hcat(Pr,Pr2)
    PP
end

function lt_pp(θ_t) # This function gives phase portrait of the transformed system from the normal form
    np1=θ_t[end];np2=3.85
#    nf=nf_dis(U₀,3.85,Vel,Vel2)
    nf=nf_dis(np1,np2,Vel,Vel2)
    vl=nf.v;vl2=nf.v2
    p1,p2,p3,p4,p5,p6=θ_t[1:6]
    T=[p1 p3;p2 p4]
    dis=transpose([p5*ones(θ_l) p6*ones(θ_l)])/scale_f_l
    vlT=[dis*norm(vl[i][:,1])^2+(T+reshape(ann_l([norm(vl[i][:,1]),Vel[i]-np1],θ_t[7:end-1]),2,2)/scale_f2)*(vl[i]) for i in 1:length(Vel)]
    vlT2=[dis*norm(vl2[i][:,1])^2+(T+reshape(ann_l([norm(vl2[i][:,1]),Vel2[i]-np1],θ_t[7:end-1]),2,2)/scale_f2)*(vl2[i]) for i in 1:length(Vel2)]
    vcat(vlT,vlT2)
end

function Array_chain(gu,ann,p) # vectorized input-> vectorized neural net
    al=length(gu[1,:])
    AC=zeros(2,0)
    for i in 1:al
        AC=hcat(AC,ann(gu[:,i],p))
    end
    AC
end
# nonlinear transformation

function loss_lt(θ_t)
    pred = predict_lt(θ_t)
    sum(abs2, AA .- pred) # + 1e-5*sum(sum.(abs, params(ann)))
end

function loss_lt2(θ_t)
    pred = predict_lt2(θ_t)
    sum(abs2, AA .- pred) # + 1e-5*sum(sum.(abs, params(ann)))
end
## Generate initial guess of the parameters (Simple linear transformation with rotation)
rot=π*0.1
R=[cos(rot) -sin(rot);sin(rot) cos(rot)]
θ=vec(1e-2*R*[8.0 0.0;0.0 1.7])
θ=vcat(θ,zeros(2))
scale_f_l=1e1 # optimization works for scale_f_l>=50 for small scale_f_l optimization does not work.

hidden=21
ann_l = FastChain(FastDense(2, hidden, tanh),FastDense(hidden, hidden, tanh), FastDense(hidden,  4))
θl = initial_params(ann_l)
scale_f2=1e2
θ=vcat(θ,θl)
pp=[18.2]
θ=vcat(θ,pp)

#θ=θ_
@time loss_lt(θ)
res_l = DiffEqFlux.sciml_train(loss_lt, θ, ADAM(0.001), maxiters = 500)
res_l = DiffEqFlux.sciml_train(loss_lt, res_l.minimizer, BFGS(initial_stepnorm=1e-4), maxiters = 10000)
U₀=res_l.minimizer[end]
res_l = DiffEqFlux.sciml_train(loss_lt2, res_l.minimizer, BFGS(initial_stepnorm=1e-4), maxiters = 30000)

res_l.minimum
θ_=res_l.minimizer
# Check the phase portrait of the linear transformation to see transformation is working properly
Ap=lt_pp(θ_)
ind=1
plot(Ap[ind][1,:],Ap[ind][2,:],xlabel="Heave (m)",ylabel="Pitch (rad)",label="ML model (U=15.5 stable LCO)")
plot!(t_series[ind][1,:],t_series[ind][2,:],label="Data (U=15.5 stable LCO)",seriestype = :scatter,markersize=1.5,markerstrokewidth=0)
ind=4
plot!(Ap[ind][1,:],Ap[ind][2,:],label="ML model (U=18.0 stable LCO)")
plot!(t_series[ind][1,:],t_series[ind][2,:],label="Data (U=18.0 stable LCO)",legend=:topleft,seriestype = :scatter,markersize=1.5,markerstrokewidth=0)

@pgf Axis( {xlabel=L"$h$ (m)",
            ylabel = L"$\alpha$ (rad)",
            legend_pos  = "north west",
            height="9cm",
            width="9cm"},

    Plot(
        { color="red",
            no_marks
        },
        Coordinates(Ap[1][1,:],Ap[1][2,:])
    ),
    LegendEntry("Model"),
    Plot(
        { color="blue",
            no_marks,
        },
        Coordinates(t_series[1][1,:],t_series[1][2,:])
    ),
    LegendEntry("Measured data")
)


savefig("PP_compare_LT.pdf")

## Add neural network to transformation to improve the model
function predict_nt(θ_t)
    p1,p2,p3,p4,p5,p6=θ_[1:6]
    T=[p1 p3;p2 p4]
    dis=transpose([p5*ones(θ_l) p6*ones(θ_l)])/scale_f_l
    np1=θ_[end];np2=3.85
    pn=θ_t
    U₀=np1
    T=[p1 p3;p2 p4]

    nf=nf_dis(np1,np2,Vel,Vel2)
    vl=nf.v;vl2=nf.v2

    vlT=[dis*norm(vl[i][:,1])^2+(T+reshape(ann_l([norm(vl[i][:,1]),Vel[i]-U₀],θ_[7:end-1]),2,2)/scale_f2)*(vl[i])+Array_chain([vl[i];(Vel[i]-np1)*ones(1,θ_l)],ann,pn)/scale_f for i in 1:length(Vel)]
    vlT2=[dis*norm(vl2[i][:,1])^2+(T+reshape(ann_l([norm(vl2[i][:,1]),Vel[i]-U₀],θ_[7:end-1]),2,2)/scale_f2)*(vl2[i])+Array_chain([vl2[i];(Vel2[i]-np1)*ones(1,θ_l)],ann,pn)/scale_f for i in 1:length(Vel2)]

    Pr=f_coeff(vlT,Vel,0,0)
    Pr2=f_coeff(vlT2,Vel2,0,0)
    hcat(Pr,Pr2)
end

function lt_pp_n(θ_t) # This function gives phase portrait of the transformed system from the normal form (stable LCO)
    p1,p2,p3,p4,p5,p6=θ_[1:6]
    T=[p1 p3;p2 p4]
    dis=transpose([p5*ones(θ_l) p6*ones(θ_l)])/scale_f_l
    np1=θ_[end];np2=3.85
    pn=θ_t
    U₀=np1
    T=[p1 p3;p2 p4]

    nf=nf_dis(np1,np2,Vel,Vel2)
    vl=nf.v;vl2=nf.v2

    vlT=[dis*norm(vl[i][:,1])^2+(T+reshape(ann_l([norm(vl[i][:,1]),Vel[i]-U₀],θ_[7:end-1]),2,2)/scale_f2)*(vl[i])+Array_chain([vl[i];(Vel[i]-np1)*ones(1,θ_l)],ann,pn)/scale_f for i in 1:length(Vel)]
    vlT2=[dis*norm(vl2[i][:,1])^2+(T+reshape(ann_l([norm(vl2[i][:,1]),Vel[i]-U₀],θ_[7:end-1]),2,2)/scale_f2)*(vl2[i])+Array_chain([vl2[i];(Vel2[i]-np1)*ones(1,θ_l)],ann,pn)/scale_f for i in 1:length(Vel2)]
    vcat(vlT,vlT2)
end


function loss_nt(θ_t)
    pred = predict_nt(θ_t)
    sum(abs2, AA .- pred) # + 1e-5*sum(sum.(abs, params(ann)))
end


function nf_dis2(p,Vel,Vel2,θ₀,θu₀,t_info)
    p1,p2,p3,p4,p5,p6=θ_[1:6]
    T=[p1 p3;p2 p4]
    U₀,s=θ_2[1:2]
    pn=θ_2[3:end]
    t_end,t_l=t_info
    p1=p[1]
    p2=p[2:end]
    del=Vel-U₀*ones(length(Vel))
    del2=Vel2-U₀*ones(length(Vel2))
    va2=s*ones(length(Vel))
    va2_2=s*ones(length(Vel2))
    s_amp=sqrt.(va2/2+sqrt.(va2.^2+4*del)/2)
    u_amp=sqrt.(va2_2/2-sqrt.(va2_2.^2+4*del2)/2)

    t=range(0, stop = t_end, length = t_l)
    ω₀=p1
    om=[ω₀+norm(ann2(s_amp[i],p2)) for i in 1:length(Vel)]
    omu=[ω₀+norm(ann2(u_amp[i],p2)) for i in 1:length(Vel2)]
    θ=[θ₀[j]*ones(length(t))+om[j]*t for j in 1:length(Vel)]
    dis=transpose([p5*ones(length(t)) p6*ones(length(t))])/scale_f_l
    θu=[θu₀[j]*ones(length(t))+omu[j]*t for j in 1:length(Vel2)]

    coθ=[cos.(θ[i]) for i in 1:length(Vel)]
    siθ=[sin.(θ[i]) for i in 1:length(Vel)]
    coθu=[cos.(θu[i]) for i in 1:length(Vel2)]
    siθu=[sin.(θu[i]) for i in 1:length(Vel2)]
    vl=[s_amp[i]*[coθ[i]';siθ[i]'] for i in 1:length(Vel)]
    vl2=[u_amp[i]*[coθu[i]';siθu[i]'] for i in 1:length(Vel2)]
    vlT=[dis*norm(vl[i][:,1])^2+T*(vl[i])+Array_chain([vl[i];(Vel[i]-U₀)*ones(1,length(t))],ann,pn)/scale_f for i in 1:length(Vel)]
    vlT2=[dis*norm(vl2[i][:,1])^2+T*(vl2[i])+Array_chain([vl2[i];(Vel2[i]-U₀)*ones(1,length(t))],ann,pn)/scale_f for i in 1:length(Vel2)]
    (v=vlT,v2=vlT2)
end

hidden=21
ann = FastChain(FastDense(3, hidden, tanh),FastDense(hidden, hidden, tanh), FastDense(hidden,  2))
θn = initial_params(ann)
scale_f=1e3

loss_nt(θn)

res_l = DiffEqFlux.sciml_train(loss_lt2, res_l.minimizer, BFGS(initial_stepnorm=1e-4), maxiters = 100000)
θ_=res_l.minimizer
res1 = DiffEqFlux.sciml_train(loss_nt, θn, ADAM(0.0001), maxiters = 300)
res_n = DiffEqFlux.sciml_train(loss_nt, res1.minimizer, BFGS(initial_stepnorm=1e-4), maxiters = 10000)
#res_n2 = DiffEqFlux.sciml_train(loss_nt, res_n.minimizer, BFGS(initial_stepnorm=1e-3), maxiters = 20000)


res_n.minimum
θ_n=res_n.minimizer
Ap=lt_pp_n(θ_n)
#Checking the phase portrait of the model (Stable LCO)
ind=1
Vel[ind]
plot(Ap[ind][1,:],Ap[ind][2,:],xlabel="Heave (m)",ylabel="Pitch (rad)",label="ML model (U=15.5 stable LCO)")
plot!(t_series[ind][1,:],t_series[ind][2,:],label="Data (U=15.5 stable LCO)",seriestype = :scatter,markersize=1.5,markerstrokewidth=0)
ind=7
plot!(Ap[ind][1,:],Ap[ind][2,:],label="ML model (U=18.0 stable LCO)")
plot!(t_series[ind][1,:],t_series[ind][2,:],label="Data (U=18.0 stable LCO)",legend=:topleft,seriestype = :scatter,markersize=1.5,markerstrokewidth=0)

@pgf Axis( {xlabel=L"$h$ (m)",
            ylabel = L"$\alpha$ (rad)",
            legend_pos  = "north west",
            height="9cm",
            width="9cm"},

    Plot(
        { color="red",
            no_marks
        },
        Coordinates(Ap[1][1,:],Ap[1][2,:])
    ),
    LegendEntry("Model"),
    Plot(
        { color="blue",
            no_marks,
        },
        Coordinates(t_series[1][1,:],t_series[1][2,:])
    ),
    LegendEntry("Measured data")
)


savefig("PP_compare_stable.pdf")
#Checking the phase portrait of the model (Unstable LCO)
Ap=lt_pp_n_u(θ_n,Vel2)
ind=1 # Near the fold
vv=Vel2[ind]
plot(Ap[ind][1,:],Ap[ind][2,:],xlabel="Heave (m)",ylabel="Pitch (rad)",label="Model (U= $(@sprintf("%.2f", vv))  unstable LCO)")
plot!(t_series[ind+4][1,:],t_series[ind+4][2,:],label="Data (U=18.0 stable LCO)",legend=:topleft,seriestype = :scatter,markersize=1.5,markerstrokewidth=0)

ind=3 # Near the fold
vv=Vel2[ind]
plot(Ap[ind][1,:],Ap[ind][2,:],xlabel="Heave (m)",ylabel="Pitch (rad)",label="Model (U= $(@sprintf("%.2f", vv))  unstable LCO)")
plot!(t_series[ind+4][1,:],t_series[ind+4][2,:],label="Data (U=18.0 stable LCO)",legend=:topleft,seriestype = :scatter,markersize=1.5,markerstrokewidth=0)


@pgf Axis( {xlabel=L"$h$ (m)",
            ylabel = L"$\alpha$ (rad)",
            legend_pos  = "north west",
            height="9cm",
            width="9cm",
            xmin=-1e-2,xmax=1e-2,ymin=2.5e-2,ymin=-3.5e-2,
            xtick=-1e-2:4e-3:1e-2},

    Plot(
        { color="red",
            no_marks
        },
        Coordinates(Ap[1][1,:],Ap[1][2,:])
    ),
    LegendEntry("Model"),
    Plot(
        { color="blue",
            no_marks,
        },
        Coordinates(vcat(uu.u[1,:],uu.u[1,1]),vcat(uu.u[3,:],uu.u[3,1]))
    ),
    LegendEntry("Measured data")
)

ind=2
vv=Vel2[ind]# Near the equilibrium
plot!(Ap[ind][1,:],Ap[ind][2,:],label="Model (U=$(@sprintf("%.2f", vv))  unstable LCO)")
uu=get_sol(U[s_ind[ind]],50,1,3)
plot!(uu.u[1,:],uu.u[3,:],label="Data (U=$(@sprintf("%.2f", vv))  unstable LCO)",seriestype = :scatter,markersize=3,legend=:bottomright)
savefig("PP_compare_u.pdf")

@pgf Axis( {xlabel=L"$h$ (m)",
            ylabel = L"$\alpha$ (rad)",
            legend_pos  = "north west",
            height="9cm",
            width="9cm",
            xmin=-1e-2,xmax=1e-2,ymin=2.5e-2,ymin=-3.5e-2,
            xtick=-1e-2:4e-3:1e-2},

    Plot(
        { color="red",
            no_marks
        },
        Coordinates(Ap[2][1,:],Ap[2][2,:])
    ),
    LegendEntry("Model"),
    Plot(
        { color="blue",
            no_marks,
        },
        Coordinates(vcat(uu.u[1,:],uu.u[1,1]),vcat(uu.u[3,:],uu.u[3,1]))
    ),
    LegendEntry("Measured data")
)

u_ind=[400,750]
Vel3=[P[u_ind[i]] for i in 1:length(u_ind)]
Ap=lt_pp_n_u(θ_2,Vel3)

ind=1
vv=Vel3[ind]
plot(Ap[ind][1,:],Ap[ind][2,:],xlabel="Heave (m)",ylabel="Pitch (rad)",label="Model (U=$(@sprintf("%.2f", vv))  unstable LCO)")
uu=get_sol(U[u_ind[ind]],N,1,3)
plot!(uu.u[1,:],uu.u[3,:],label="Data (U=$(@sprintf("%.2f", vv))  unstable LCO)",seriestype = :scatter,markersize=3)

@pgf Axis( {xlabel=L"$h$ (m)",
            ylabel = L"$\alpha$ (rad)",
            legend_pos  = "north west",
            height="9cm",
            width="9cm",
            xmin=-1.5e-2,xmax=2e-2,ymin=4.5e-2,ymin=-6.5e-2
},

    Plot(
        { color="red",
            no_marks
        },
        Coordinates(Ap[1][1,:],Ap[1][2,:])
    ),
    LegendEntry("Model"),
    Plot(
        { color="blue",
            no_marks,
        },
        Coordinates(vcat(uu.u[1,:],uu.u[1,1]),vcat(uu.u[3,:],uu.u[3,1]))
    ),
    LegendEntry("Measured data")
)

ind=2 # Near the equilibrium
vv=Vel3[ind]
plot!(Ap[ind][1,:],Ap[ind][2,:],label="Model (U=$(@sprintf("%.2f", vv))  unstable LCO)")
uu=get_sol(U[u_ind[ind]],50,1,3)
plot!(uu.u[1,:],uu.u[3,:],label="Data (U=$(@sprintf("%.2f", vv))  unstable LCO)",seriestype = :scatter,markersize=3,legend=:bottomright)

@pgf Axis( {xlabel=L"$h$ (m)",
            ylabel = L"$\alpha$ (rad)",
            legend_pos  = "north east",
            height="9cm",
            width="9cm",
            xmin=-1.5e-2,xmax=2e-2,ymin=4.5e-2,ymin=-6.5e-2
},

    Plot(
        { color="red",
            no_marks
        },
        Coordinates(Ap[2][1,:],Ap[2][2,:])
    ),
    LegendEntry("Model"),
    Plot(
        { color="blue",
            no_marks,
        },
        Coordinates(vcat(uu.u[1,:],uu.u[1,1]),vcat(uu.u[3,:],uu.u[3,1]))
    ),
    LegendEntry("Measured data")
)

savefig("PP_compare_u2.pdf")

θ=[θ_;θ_n]
θ_t=θ
## Compare the bifurcation diagram
function lt_b_dia(θ_t,ind)
    vel_l=300
    θ_=θ_t[1:620]
    p1,p2,p3,p4,p5,p6=θ_[1:6]
    T=[p1 p3;p2 p4]
    np1=θ_[end];np2=3.85
    pn=θ_t[621:end]
    U₀=np1

    Vel=range(np1-np2^2/4+1e-7, stop = np1, length = vel_l)

    dis=transpose([p5*ones(θ_l) p6*ones(θ_l)])/scale_f_l
    nf=nf_dis(np1,np2,Vel,Vel)
    vl=nf.v;vl2=nf.v2

    vlT=[dis*norm(vl[i][:,1])^2+(T+reshape(ann_l([norm(vl[i][:,1]),Vel[i]-U₀],θ_[7:end-1]),2,2)/scale_f2)*(vl[i])+Array_chain([vl[i];(Vel[i]-np1)*ones(1,θ_l)],ann,pn)/scale_f for i in 1:length(Vel)]
    vlT2=[dis*norm(vl2[i][:,1])^2+(T+reshape(ann_l([norm(vl2[i][:,1]),Vel[i]-U₀],θ_[7:end-1]),2,2)/scale_f2)*(vl2[i])+Array_chain([vl2[i];(Vel[i]-np1)*ones(1,θ_l)],ann,pn)/scale_f for i in 1:length(Vel)]

    vlTas=[maximum(vlT[i][ind,:])-minimum(vlT[i][ind,:]) for i in 1:length(Vel)]
    vlTau=[maximum(vlT2[i][ind,:])-minimum(vlT2[i][ind,:]) for i in 1:length(Vel)]

    return (s=vlTas,u=vlTau,v=Vel)
end

bd=lt_b_dia(θ_t,1)
h=[maximum(t_series[i][1,:])-minimum(t_series[i][1,:]) for i in 1:length(Vel)]
h2=[maximum(t_series[i+4][1,:])-minimum(t_series[i+4][1,:]) for i in 1:length(Vel2)]
d_amp=[amp[s_ind[i]] for i in 1:length(s_ind)]
d_P=[P[s_ind[i]] for i in 1:length(s_ind)]

plot(bd.v,bd.s,label="Stable LCO (ML model)")
plot!(bd.v,bd.u,label="Unstable LCO (ML model)")
plot!(Vel,h,seriestype = :scatter,label="Training data (stable)",legend=:right,xlabel="Wind speed (m/sec)",ylabel="Heave amplitude (m)",markerstrokewidth=0)
plot!(Vel2,h2,seriestype = :scatter,label="Training data (stable)",legend=:right,xlabel="Wind speed (m/sec)",ylabel="Heave amplitude (m)",markerstrokewidth=0)

savefig("BD_compare.pdf")
P=vec(P)
amp=vec(amp)

vv=vcat(bd.v,bd.v)
aa=vcat(bd.s,bd.u)

@pgf Axis( {xlabel="Wind speed (m/sec)",
            ylabel = "Heave amplitude (m)",
            legend_pos  = "north west",
            height="11cm",
            width="15cm",
            ymin=0,ymax=9e-2,
            mark_options = {scale=1.5}
},
Plot(
    { color="blue",
        only_marks,
    },
    Coordinates(Vel,h)
),
    LegendEntry("Training data  (stable LCO)"),
    Plot(
        { color="red",
            only_marks,
            mark = "triangle*"
        },
        Coordinates(d_P,d_amp)
    ),
    LegendEntry("Training data (unstable LCO)"),

    Plot(
        { color="red",
            no_marks,
        },
        Coordinates(P,amp)
    ),
    LegendEntry("Numerical continuation of Experimental model"),
    Plot(
        { color="blue",
            no_marks
        },
        Coordinates(bd.v,bd.s)
    ),
    LegendEntry("ML model"),

    Plot(
        { color="blue",
            no_marks,
        },
        Coordinates(bd.v,bd.u)
    ),
)

## Speed of phase

function Inv_T_u(th0,vel,tol) # This function gives phase portrait of the transformed system from the normal form (unstable LCO)
    θ_=θ_t[1:620]
    p1,p2,p3,p4,p5,p6=θ_[1:6]
    T=[p1 p3;p2 p4]
    np1=θ_[end];np2=3.85
    pn=θ_t[621:end]
    U₀=np1

    s_amp=sqrt(np2/2+sqrt(np2^2+4*(vel-np1))/2)
    theta=range(-π,stop=π,length=300)

    uu=[[s_amp*cos(theta[i]),s_amp*sin(theta[i])] for i in 1:length(theta)]
    u=[[p5,p6]*norm(uu[i])^2/scale_f_l+(T+reshape(ann_l([norm(uu[i][:,1]),vel-U₀],θ_[7:end-1]),2,2)/scale_f2)*uu[i]+ann([uu[i];vel-np1],pn)/scale_f for i in 1:length(theta)]
    t0=[abs(atan(u[i][2],u[i][1])-th0) for i in 1:length(theta)]
    er=minimum(t0)
    while er>tol
    #    global theta,t0
        theta=range(theta[argmin(t0)-1],theta[argmin(t0)+1],length=300)
        uu=[[s_amp*cos(theta[i]),s_amp*sin(theta[i])] for i in 1:length(theta)]
        u=[[p5,p6]*norm(uu[i])^2/scale_f_l+(T+reshape(ann_l([norm(uu[i][:,1]),vel-U₀],θ_[7:end-1]),2,2)/scale_f2)*uu[i]+ann([uu[i];vel-np1],pn)/scale_f for i in 1:length(theta)]
        t0=[abs(atan(u[i][2],u[i][1])-th0) for i in 1:length(theta)]
        er=minimum(t0)
    end
    return     theta[argmin(t0)]
end

function dudt_nf(u,p,t)
    c=u[3]
    np1,np2=θ_2[1:2]
    a2=np2;δ₀=np1
    ν=(c-δ₀)
    r2=u[1]^2+u[2]^2
    ω₀=p[1]
    uu=[u[1],u[2],ν]
    ph=ω₀+ann3(uu,p[2:end])[1]/om_scale
    du₁=ν*u[1]-u[2]*ph+a2*u[1]*r2-u[1]*r2^2
    du₂=u[1]*ph+ν*u[2]+a2*u[2]*r2-u[2]*r2^2
    du₃=0
    [du₁,du₂,du₃]
end

function dudt_ph(u,p,t)
    θ_=θ_t[1:620]
    np1=θ_[end];np2=3.85

    θ=u[1]
    r=u[2]
    c=u[3]
    a2=np2;δ₀=np1
    ν=(c-δ₀)
    ω₀=p[1]
    uu=[r*cos(θ),r*sin(θ),ν]
    du₁=ω₀+ann3(uu,p[2:end])[1]/om_scale
    du₂=0
    du₃=0
    [du₁,du₂,du₃]
end

function predict_time_T(p) #,uu_t0
    θ_=θ_t[1:620]
    p1,p2,p3,p4,p5,p6=θ_[1:6]
    T=[p1 p3;p2 p4]
    np1=θ_[end];np2=3.85
    pn=θ_t[621:end]
    U₀=np1

    A1=[Array(concrete_solve(ODEProblem(dudt_ph,u_t0[i],(0,tl2),p), Tsit5(), u_t0[i], p, saveat = st,
                         abstol=1e-8, reltol=1e-8,
                         sensealg = InterpolatingAdjoint(autojacvec=ReverseDiffVJP()))) for i in 1:length(Vel)]
    uu=[transpose(hcat(A1[i][2,:].*cos.(A1[i][1,:]),A1[i][2,:].*sin.(A1[i][1,:]),A1[i][3,:])) for i in 1:vl]
    delU=zeros(2,spl)
    delU2=-np1*ones(1,spl)
    delU=vcat(delU,delU2)
    uu=[uu[i]+delU for i in 1:vl]
    dis=transpose([p5*ones(spl) p6*ones(spl)])/scale_f_l

    #vlT=[dis*norm(uu[i][1:2])^2+T*uu[i][1:2,:]+Array_chain(uu[i],ann,pn)/scale_f for i in 1:vl]
    vlT=[dis*norm(uu[i][1:2,1])^2+(T+reshape(ann_l([norm(uu[i][1:2,1]),Vel[i]-U₀],θ_[7:end-1]),2,2)/scale_f2)*(uu[i][1:2,:])+Array_chain(uu[i],ann,pn)/scale_f for i in 1:length(Vel)]

    Pr=zeros(0,spl)
    for i in 1:vl
        theta=vlT[i][[1,2],:]
        Pr=vcat(Pr,theta)
    end
Pr
end

tl2=1.0
spl=Int(tl2/st+1)
spl=1001
st=1e-3
vl=length(Vel)
hidden=31
ann3 = FastChain(FastDense(3, hidden, tanh),FastDense(hidden, 1, tanh))
np = initial_params(ann3)
omega=15.3
p = vcat(omega,np)
tol=1e-5


# Generate data and initial θ
np1=θ_t[620]
np2=3.85
pn=θ_2[3:end]
s_amp=[sqrt(np2/2+sqrt(np2^2+4*(Vel[i]-np1))/2) for i in 1:length(Vel)]
u_amp=[sqrt(np2/2-sqrt(np2^2+4*(Vel2[i]-np1))/2) for i in 1:length(Vel2)]
theta0=[atan(t_series[i][2,1],t_series[i][1,1]) for i in 1:length(Vel)]
θ₀=[Inv_T_u(theta0[i],Vel[i],tol) for i in 1:length(Vel)]
u_t0=[[θ₀[i],s_amp[i],Vel[i]] for i in 1:length(Vel)]
u_t02=[[s_amp[i]*cos.(θ₀[i]),s_amp[i]*sin.(θ₀[i]),Vel[i]] for i in 1:length(Vel)]

uu_t0=[[θ₀[i],u_amp[i],Vel2[i]] for i in 1:length(Vel2)]
uu_t02=[[u_amp[i]*cos.(θ₀[i]),u_amp[i]*sin.(θ₀[i]),Vel2[i]] for i in 1:length(Vel2)]

spl=length(t_series[1][1,:])
t_s=zeros(vl*2,spl)

for i in 1:vl
    t_s[[2*(i-1)+1,2*(i-1)+2],:]=t_series[i][:,1:1000]
end
A3=t_s
t_series

function loss_time_T(p)
    pred = predict_time_T(p)
#    pred = hcat(pred)
    sum(abs2, A3 .- pred) # + 1e-5*sum(sum.(abs, params(ann)))
#    norm(pred-A3)
end

om_scale=0.3
loss_time_T(p)
res1 = DiffEqFlux.sciml_train(loss_time_T, p, ADAM(0.01), maxiters = 300)
res1 = DiffEqFlux.sciml_train(loss_time_T, res1.minimizer, BFGS(initial_stepnorm=1e-3), maxiters = 10000)

res1.minimum
p=res1.minimizer

@save "flutter.jld" p

tv=range(0,1,length=1001)
ind=1
plot(tv,predict_time_T(p)[2*(ind-1)+1,:],xlims=(0.0,1.0),xlabel="time (sec)", ylabel="Heave (m)",label="Model U = 15.0 m/sec")
plot!(tv,t_series[ind][1,:],label="data",seriestype = :scatter,markersize=2,markerstrokewidth=0)

ind=10
plot!(tv,predict_time_T(p)[2*(ind-1)+1,:],xlims=(0.0,1.0),xlabel="time (sec)", ylabel="Heave (m)",label="Model U = 18.0 m/sec")
plot!(tv,t_series[ind][1,:],label="data",seriestype = :scatter,markersize=2,markerstrokewidth=0)

@pgf Axis( {xlabel="Time (sec)",
            ylabel = L"$h$ (m)",
            legend_pos  = "north west",
            height="9cm",
            width="9cm",xmin=0,xmax=1,ymax=6e-2,ymin=-4e-2,mark_options = {scale=0.1}},

    Plot(
        { color="red",
            no_marks
        },
        Coordinates(tv,predict_time_T(p)[2*(1-1)+1,:])
    ),
    LegendEntry("Model"),
    Plot(
        { color="blue",
            no_marks,
        },
        Coordinates(tv,t_series[1][1,:])
    ),
    LegendEntry("Measured data")
)


@pgf Axis( {xlabel="Time (sec)",
            ylabel = L"$h$ (m)",
            legend_pos  = "north west",
            height="9cm",
            width="9cm",xmin=0,xmax=1,ymax=6e-2,ymin=-4e-2},

    Plot(
        { color="red",
            no_marks
        },
        Coordinates(tv,predict_time_T(p)[2*(10-1)+1,:])
    ),
    LegendEntry("Model"),
    Plot(
        { color="blue",
            no_marks,
        },
        Coordinates(tv,t_series[10][1,:])
    ),
    LegendEntry("Measured data")
)

savefig("freq.pdf")

@save "flutter.jld"
