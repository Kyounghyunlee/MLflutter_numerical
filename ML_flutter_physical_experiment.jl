# Training ML model with experimental data

using OrdinaryDiffEq
using ModelingToolkit
using DataDrivenDiffEq
using LinearAlgebra, DiffEqSensitivity, Optim, Statistics
using DiffEqFlux, Flux
using Printf,PGFPlotsX,LaTeXStrings, JLD2
using MAT
include("Numerical_Cont.jl")

## Save data
nh=30
l=6000
vars = matread("./measured_data/CBC_stable_v14_9.mat")
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

vars = matread("./measured_data/CBC_stable_v15_6.mat")
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

vars = matread("./measured_data/CBC_stable_v16_5.mat")
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

vars = matread("./measured_data/CBC_stable_v17_3.mat")
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

vars = matread("./measured_data/CBC_unstable_v14_9.mat")
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

vars = matread("./measured_data/CBC_unstable_v15_6.mat")
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

vars = matread("./measured_data/CBC_unstable_v16_5.mat")
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
        rN=Transpose(tM)*r
        c=inv(MM)*rN
        Pr=hcat(Pr,c)
        Pr
    end
    Pr
end


function predict_lt(θ_t) #predict the linear transformation
    np1=θ_t[end-1];np2=θ_t[end]
    nf=nf_dis(np1,np2,Vel,Vel2)
    vl=nf.v;vl2=nf.v2
    p1,p2,p3,p4,p5,p6=θ_t[1:6]
    T=[p1 p3;p2 p4]

    dis=transpose([p5*ones(θ_l) p6*ones(θ_l)])/scale_f_l
    pn=θ_t
    vlT=[dis*norm(vl[i][:,1])^2+(T+reshape(ann_l([norm(vl[i][:,1]),Vel[i]-np1],θ_t[7:end-2]),2,2)/scale_f2)*(vl[i]) for i in 1:length(Vel)]
    vlT2=[dis*norm(vl2[i][:,1])^2+(T+reshape(ann_l([norm(vl2[i][:,1]),Vel2[i]-np1],θ_t[7:end-2]),2,2)/scale_f2)*(vl2[i]) for i in 1:length(Vel2)]

    Pr=f_coeff(vlT,Vel,0,0)
    Pr2=f_coeff(vlT2,Vel2,0,0)
    PP=hcat(Pr,Pr2)
    PP
end

function predict_lt2(θ_t) #predict the linear transformation
    np1=U₀;np2=s_
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
    np1=U₀;np2=s_
    nf=nf_dis(np1,np2,Vel,Vel2)
    vl=nf.v;vl2=nf.v2
    p1,p2,p3,p4,p5,p6=θ_t[1:6]
    T=[p1 p3;p2 p4]
    dis=transpose([p5*ones(θ_l) p6*ones(θ_l)])/scale_f_l
    vlT=[dis*norm(vl[i][:,1])^2+(T+reshape(ann_l([norm(vl[i][:,1]),Vel[i]-np1],θ_t[7:end-2]),2,2)/scale_f2)*(vl[i]) for i in 1:length(Vel)]
    vlT2=[dis*norm(vl2[i][:,1])^2+(T+reshape(ann_l([norm(vl2[i][:,1]),Vel2[i]-np1],θ_t[7:end-2]),2,2)/scale_f2)*(vl2[i]) for i in 1:length(Vel2)]
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

hidden=12
ann_l = FastChain(FastDense(2, hidden, tanh),FastDense(hidden, hidden, tanh), FastDense(hidden,  4))
θl = initial_params(ann_l)
scale_f2=1e2
θ=vcat(θ,θl)
pp=[17.95,3.85]
θ=vcat(θ,pp) # Initial guess of parameters

## Add neural network to transformation to improve the model
function predict_nt(θ_t)
    p1,p2,p3,p4,p5,p6=θ_[1:6]
    T=[p1 p3;p2 p4]
    dis=transpose([p5*ones(θ_l) p6*ones(θ_l)])/scale_f_l
    np1=U₀;np2=s_
    pn=θ_t
    T=[p1 p3;p2 p4]

    nf=nf_dis(np1,np2,Vel,Vel2)
    vl=nf.v;vl2=nf.v2

    vlT=[dis*norm(vl[i][:,1])^2+(T+reshape(ann_l([norm(vl[i][:,1]),Vel[i]-U₀],θ_[7:end-2]),2,2)/scale_f2)*(vl[i])+Array_chain([vl[i];(Vel[i]-np1)*ones(1,θ_l)],ann,pn)/scale_f for i in 1:length(Vel)]
    vlT2=[dis*norm(vl2[i][:,1])^2+(T+reshape(ann_l([norm(vl2[i][:,1]),Vel[i]-U₀],θ_[7:end-2]),2,2)/scale_f2)*(vl2[i])+Array_chain([vl2[i];(Vel2[i]-np1)*ones(1,θ_l)],ann,pn)/scale_f for i in 1:length(Vel2)]

    Pr=f_coeff(vlT,Vel,0,0)
    Pr2=f_coeff(vlT2,Vel2,0,0)
    hcat(Pr,Pr2)
end

function lt_pp_n(θ_t) # This function gives phase portrait of the transformed system from the normal form (stable LCO)
    p1,p2,p3,p4,p5,p6=θ_[1:6]
    T=[p1 p3;p2 p4]
    dis=transpose([p5*ones(θ_l) p6*ones(θ_l)])/scale_f_l
    np1=U₀;np2=s_
    pn=θ_t
    T=[p1 p3;p2 p4]

    nf=nf_dis(np1,np2,Vel,Vel2)
    vl=nf.v;vl2=nf.v2

    vlT=[dis*norm(vl[i][:,1])^2+(T+reshape(ann_l([norm(vl[i][:,1]),Vel[i]-U₀],θ_[7:end-2]),2,2)/scale_f2)*(vl[i])+Array_chain([vl[i];(Vel[i]-np1)*ones(1,θ_l)],ann,pn)/scale_f for i in 1:length(Vel)]
    vlT2=[dis*norm(vl2[i][:,1])^2+(T+reshape(ann_l([norm(vl2[i][:,1]),Vel[i]-U₀],θ_[7:end-2]),2,2)/scale_f2)*(vl2[i])+Array_chain([vl2[i];(Vel2[i]-np1)*ones(1,θ_l)],ann,pn)/scale_f for i in 1:length(Vel2)]
    vcat(vlT,vlT2)
end

function loss_nt(θ_t) # Loss function
    pred = predict_nt(θ_t)
    sum(abs2, AA .- pred) # + 1e-5*sum(sum.(abs, params(ann)))
end

# set neural network
hidden=11
ann = FastChain(FastDense(3, hidden, tanh),FastDense(hidden, hidden, tanh), FastDense(hidden,  2))
θn = initial_params(ann)
scale_f=1e3

loss_nt(θn)

res_l = DiffEqFlux.sciml_train(loss_lt, θ, ADAM(0.001), maxiters = 800) # First, train the simple model
U₀=res_l.minimizer[end-1];s_=res_l.minimizer[end]
θ_=res_l.minimizer
res_l = DiffEqFlux.sciml_train(loss_lt2, res_l.minimizer, BFGS(initial_stepnorm=1e-4), maxiters = 20000)
res1 = DiffEqFlux.sciml_train(loss_nt, θn, ADAM(0.001), maxiters = 1000) # Train more complicated model
res_n = DiffEqFlux.sciml_train(loss_nt, res1.minimizer, BFGS(initial_stepnorm=1e-4), maxiters = 10000)

θ_n=res_n.minimizer
Ap=lt_pp_n(θ_n)

sens=18/100 # sensitivity of the lasor sensor

a=@pgf Axis( {xlabel=L"$h$ (m)",
            ylabel = L"$\alpha$ (rad)",
            legend_pos  = "north west",
            height="9cm",
            width="9cm",
            xmin=-4e-2,xmax=4e-2,ymax=7e-2,ymin=-8e-2},

    Plot(
        { color="red",
            no_marks
        },
        Coordinates(Ap[1][1,:]*sens,Ap[1][2,:])
    ),
    LegendEntry("Learnt model"),
    Plot(
        { color="blue",
            mark  = "o",
            mark_size="0.5pt"
        },
        Coordinates(t_series[1][1,:]*sens,t_series[1][2,:])
    ),
    LegendEntry("Measured data")
)
pgfsave("./Figures/exp/pp_s149.pdf",a)

a=@pgf Axis( {xlabel=L"$h$ (m)",
            ylabel = L"$\alpha$ (rad)",
            legend_pos  = "north west",
            height="9cm",
            width="9cm",
            xmin=-4e-2,xmax=4e-2,ymax=7e-2,ymin=-8e-2},

    Plot(
        { color="red",
            no_marks
        },
        Coordinates(Ap[5][1,:]*sens,Ap[5][2,:])
    ),
    LegendEntry("Learnt model"),
    Plot(
        { color="blue",
            mark  = "o",
            mark_size="0.5pt"
        },
        Coordinates(t_series[5][1,:]*sens,t_series[5][2,:])
    ),
    LegendEntry("Measured data")
)
pgfsave("./Figures/exp/pp_s149pp_u149.pdf",a)

##
a=@pgf Axis( {xlabel=L"$h$ (m)",
            ylabel = L"$\alpha$ (rad)",
            legend_pos  = "north west",
            height="9cm",
            width="9cm",
            xmin=-4e-2,xmax=4e-2,ymax=7e-2,ymin=-8e-2},

    Plot(
        { color="red",
            no_marks
        },
        Coordinates(Ap[3][1,:]*sens,Ap[3][2,:])
    ),
    LegendEntry("Learnt model"),
    Plot(
        { color="blue",
            mark  = "o",
            mark_size="0.5pt"
        },
        Coordinates(t_series[3][1,:]*sens,t_series[3][2,:])
    ),
    LegendEntry("Measured data")
)
pgfsave("./Figures/exp/pp_s165.pdf",a)
a=@pgf Axis( {xlabel=L"$h$ (m)",
            ylabel = L"$\alpha$ (rad)",
            legend_pos  = "north west",
            height="9cm",
            width="9cm",
            xmin=-4e-2,xmax=4e-2,ymax=7e-2,ymin=-8e-2},

    Plot(
        { color="red",
            no_marks
        },
        Coordinates(Ap[7][1,:]*sens,Ap[7][2,:])
    ),
    LegendEntry("Learnt model"),
    Plot(
        { color="blue",
            mark  = "o",
            mark_size="0.5pt"
        },
        Coordinates(t_series[7][1,:]*sens,t_series[7][2,:])
    ),
    LegendEntry("Measured data")
)
pgfsave("./Figures/exp/pp_u165.pdf",a)
##

θ=[θ_;θ_n]
θ_t=θ
ll=length(θ_)
## Plot the bifurcation diagram
function lt_b_dia(θ_t,ind)
    vel_l=300
    θ_=θ_t[1:ll]
    p1,p2,p3,p4,p5,p6=θ_[1:6]
    T=[p1 p3;p2 p4]
    np1=U₀;np2=s_
    pn=θ_t[ll+1:end]
    Vel=range(np1-np2^2/4+1e-7, stop = np1, length = vel_l)

    dis=transpose([p5*ones(θ_l) p6*ones(θ_l)])/scale_f_l
    nf=nf_dis(np1,np2,Vel,Vel)
    vl=nf.v;vl2=nf.v2

    vlT=[dis*norm(vl[i][:,1])^2+(T+reshape(ann_l([norm(vl[i][:,1]),Vel[i]-U₀],θ_[7:end-2]),2,2)/scale_f2)*(vl[i])+Array_chain([vl[i];(Vel[i]-np1)*ones(1,θ_l)],ann,pn)/scale_f for i in 1:length(Vel)]
    vlT2=[dis*norm(vl2[i][:,1])^2+(T+reshape(ann_l([norm(vl2[i][:,1]),Vel[i]-U₀],θ_[7:end-2]),2,2)/scale_f2)*(vl2[i])+Array_chain([vl2[i];(Vel[i]-np1)*ones(1,θ_l)],ann,pn)/scale_f for i in 1:length(Vel)]

    vlTas=[maximum(vlT[i][ind,:])-minimum(vlT[i][ind,:]) for i in 1:length(Vel)]
    vlTau=[maximum(vlT2[i][ind,:])-minimum(vlT2[i][ind,:]) for i in 1:length(Vel)]

    return (s=vlTas,u=vlTau,v=Vel)
end

## Train the speed of phase
function Inv_T_u(th0,vel,tol) # This function computes the initial condition of the model at stable LCO
    vel_l=300
    θ_=θ_t[1:ll]
    p1,p2,p3,p4,p5,p6=θ_[1:6]
    T=[p1 p3;p2 p4]
    np1=U₀;np2=s_
    pn=θ_t[ll+1:end]

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

function Inv_T_uu(th0,vel,tol) # This function computes the initial condition of the model at unstable LCO
    vel_l=300
    θ_=θ_t[1:ll]
    p1,p2,p3,p4,p5,p6=θ_[1:6]
    T=[p1 p3;p2 p4]
    np1=U₀;np2=s_
    pn=θ_t[ll+1:end]

    s_amp=sqrt(np2/2-sqrt(np2^2+4*(vel-np1))/2)
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

function dudt_ph(u,p,t) # speed of phase
    np1=U₀;np2=s_
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

function predict_time_T(p) # Predict the time series of the model with computed initial conditions
    vel_l=300
    θ_=θ_t[1:ll]
    p1,p2,p3,p4,p5,p6=θ_[1:6]
    T=[p1 p3;p2 p4]
    np1=U₀;np2=s_
    pn=θ_t[ll+1:end]

    A1=[Array(concrete_solve(ODEProblem(dudt_ph,u_t0[i],(0,tl2),p), Tsit5(), u_t0[i], p, saveat = st,
                         abstol=1e-8, reltol=1e-8,
                         sensealg = InterpolatingAdjoint(autojacvec=ReverseDiffVJP()))) for i in 1:length(Vel)]
    uu=[transpose(hcat(A1[i][2,:].*cos.(A1[i][1,:]),A1[i][2,:].*sin.(A1[i][1,:]),A1[i][3,:])) for i in 1:length(Vel)]
    delU=zeros(2,spl)
    delU2=-np1*ones(1,spl)
    delU=vcat(delU,delU2)
    uu=[uu[i]+delU for i in 1:length(Vel)]
    dis=transpose([p5*ones(spl) p6*ones(spl)])/scale_f_l


    vlT=[dis*norm(uu[i][1:2,1])^2+(T+reshape(ann_l([norm(uu[i][1:2,1]),Vel[i]-U₀],θ_[7:end-1]),2,2)/scale_f2)*(uu[i][1:2,:])+Array_chain(uu[i],ann,pn)/scale_f for i in 1:length(Vel)]

    Pr=zeros(0,spl)
    for i in 1:length(Vel)
        theta=vlT[i][[1,2],:]
        Pr=vcat(Pr,theta)
    end
##
A1=[Array(concrete_solve(ODEProblem(dudt_ph,uu_t0[i],(0,tl2),p), Tsit5(), uu_t0[i], p, saveat = st,
                     abstol=1e-8, reltol=1e-8,
                     sensealg = InterpolatingAdjoint(autojacvec=ReverseDiffVJP()))) for i in 1:length(Vel2)]
uu=[transpose(hcat(A1[i][2,:].*cos.(A1[i][1,:]),A1[i][2,:].*sin.(A1[i][1,:]),A1[i][3,:])) for i in 1:length(Vel2)]
delU=zeros(2,spl)
delU2=-np1*ones(1,spl)
delU=vcat(delU,delU2)
uu=[uu[i]+delU for i in 1:1:length(Vel2)]
dis=transpose([p5*ones(spl) p6*ones(spl)])/scale_f_l

vlT=[dis*norm(uu[i][1:2,1])^2+(T+reshape(ann_l([norm(uu[i][1:2,1]),Vel2[i]-U₀],θ_[7:end-1]),2,2)/scale_f2)*(uu[i][1:2,:])+Array_chain(uu[i],ann,pn)/scale_f for i in 1:length(Vel2)]

Pr2=zeros(0,spl)
for i in 1:length(Vel2)
    theta=vlT[i][[1,2],:]
    Pr2=vcat(Pr2,theta)
end
##
[Pr;Pr2]
end

tl2=1.0
st=1e-3
spl=Int(tl2/st+1)
vl=length(Vel)
hidden=21
ann3 = FastChain(FastDense(3, hidden, tanh),FastDense(hidden, 1, tanh))
np = initial_params(ann3)
omega=15.3
p = vcat(omega,np)
tol=1e-5

# Generate data and initial θ
np1=U₀;np2=s_
s_amp=[sqrt(np2/2+sqrt(np2^2+4*(Vel[i]-np1))/2) for i in 1:length(Vel)]
u_amp=[sqrt(np2/2-sqrt(np2^2+4*(Vel2[i]-np1))/2) for i in 1:length(Vel2)]
theta0=[atan(t_series[i][2,1],t_series[i][1,1]) for i in 1:length(Vel)]
theta0u=[atan(t_series[i+length(Vel)][2,1],t_series[i+length(Vel)][1,1]) for i in 1:length(Vel2)]
θ₀=[Inv_T_u(theta0[i],Vel[i],tol) for i in 1:length(Vel)]
θ₀u=[Inv_T_uu(theta0u[i],Vel2[i],tol) for i in 1:length(Vel2)]
u_t0=[[θ₀[i],s_amp[i],Vel[i]] for i in 1:length(Vel)]
uu_t0=[[θ₀u[i],u_amp[i],Vel2[i]] for i in 1:length(Vel2)]

vl=7
t_s=zeros(vl*2,spl)
t_series2=[t_series[j][:,1:5:end] for j=1:7]


for i in 1:vl
    t_s[[2*(i-1)+1,2*(i-1)+2],:]=t_series2[i][:,1:1001]
end
A3=t_s

# plot bifurcation diagram
bd=lt_b_dia(θ_t,1)
h=[maximum(t_series2[i][1,:])-minimum(t_series2[i][1,:]) for i in 1:length(Vel)]
h2=[maximum(t_series2[i+4][1,:])-minimum(t_series2[i+4][1,:]) for i in 1:length(Vel2)]

a=@pgf Axis( {xlabel="Wind speed (m/sec)",
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
    Coordinates(Vel,h*sens)
),
    LegendEntry("Measured data  (stable LCO)"),
    Plot(
        { color="red",
            only_marks,
            mark = "triangle*"
        },
        Coordinates(Vel2,h2*sens)
    ),
    LegendEntry("Measured data (unstable LCO)"),

    Plot(
        { color="blue",
            no_marks
        },
        Coordinates(bd.v,bd.s*sens)
    ),
    LegendEntry("Learnt model"),

    Plot(
        { color="blue",
            no_marks,
        },
        Coordinates(bd.v,bd.u*sens)
    ),
)

pgfsave("./Figures/exp/bd.pdf",a)

function loss_time_T(p) # Loss function for the time series
    pred = predict_time_T(p)
    sum(abs2, A3 .- pred) # + 1e-5*sum(sum.(abs, params(ann)))
end

om_scale=0.3
tl2=1.0
st=1e-3
spl=Int(tl2/st+1)
vl=length(Vel)
hidden=21
ann3 = FastChain(FastDense(3, hidden, tanh),FastDense(hidden, 1, tanh))
np = initial_params(ann3)
omega=16.3
p = vcat(omega,np)
tol=1e-5

loss_time_T(p)

res_t = DiffEqFlux.sciml_train(loss_time_T, p, ADAM(0.01), maxiters = 200)
res_t = DiffEqFlux.sciml_train(loss_time_T, res_t.minimizer, BFGS(initial_stepnorm=1e-3), maxiters = 10000)

res_t.minimum
p=res_t.minimizer

tv=range(0,tl2,length=spl) #Generate a time vector
a=@pgf Axis( {xlabel="Time (sec)",
            ylabel = L"$h$ (m)",
            legend_pos  = "north west",
            height="9cm",
                width="9cm",xmin=0,xmax=1,ymax=5e-2,ymin=-5e-2,mark_options = {scale=0.1}},

    Plot(
        { color="red",
            no_marks
        },
        Coordinates(tv,sens*predict_time_T(p)[2*(1-1)+1,:])
    ),
    LegendEntry("Learnt model"),
    Plot(
        { color="blue",
            no_marks,
        },
        Coordinates(tv,sens*t_series2[1][1,1:1001])
    ),
    LegendEntry("Measured data")
)
pgfsave("./Figures/exp/t_s149.pdf",a)

a=@pgf Axis( {xlabel="Time (sec)",
            ylabel = L"$h$ (m)",
            legend_pos  = "north west",
            height="9cm",
            width="9cm",xmin=0,xmax=1,ymax=5e-2,ymin=-5e-2,mark_options = {scale=0.1}},

    Plot(
        { color="red",
            no_marks
        },
        Coordinates(tv,sens*predict_time_T(p)[2*(5-1)+1,:])
    ),
    LegendEntry("Learnt model"),
    Plot(
        { color="blue",
            no_marks,
        },
        Coordinates(tv,sens*t_series2[5][1,1:1001])
    ),
    LegendEntry("Measured data")
)

pgfsave("./Figures/exp/t_u149.pdf",a)
##

a=@pgf Axis( {xlabel="Time (sec)",
            ylabel = L"$h$ (m)",
            legend_pos  = "north west",
            height="9cm",
            width="9cm",xmin=0,xmax=1,ymax=5e-2,ymin=-5e-2,mark_options = {scale=0.1}},

    Plot(
        { color="red",
            no_marks
        },
        Coordinates(tv,sens*predict_time_T(p)[2*(3-1)+1,:])
    ),
    LegendEntry("Learnt model"),
    Plot(
        { color="blue",
        no_marks
        },
        Coordinates(tv,sens*t_series2[3][1,1:1001])
    ),
    LegendEntry("Measured data")
)
pgfsave("./Figures/exp/t_s165.pdf",a)

a=@pgf Axis( {xlabel="Time (sec)",
            ylabel = L"$h$ (m)",
            legend_pos  = "north west",
            height="9cm",
            width="9cm",xmin=0,xmax=1,ymax=5e-2,ymin=-5e-2,mark_options = {scale=0.1}},

    Plot(
        { color="red",
            no_marks
        },
        Coordinates(tv,sens*predict_time_T(p)[2*(7-1)+1,:])
    ),
    LegendEntry("Learnt model"),
    Plot(
        { color="blue",
            no_marks,
        },
        Coordinates(tv,sens*t_series2[7][1,1:1001])
    ),
    LegendEntry("Measured data")
)
pgfsave("./Figures/exp/t_u165.pdf",a)

function plot_trans(vel,a_l,amp) # plotting transformation
    vel_l=300
    θ_=θ_t[1:ll]
    p1,p2,p3,p4,p5,p6=θ_[1:6]
    T=[p1 p3;p2 p4]
    np1=U₀;np2=s_
    pn=θ_t[ll+1:end]
    Vel=vel

    s_amp=range(0.1,stop=amp,length=a_l)
    Vel=vel*ones(a_l)
    theta=range(-π,stop=π,length=θ_l)
    vl=[ones(2,300) for j=1:a_l]
    for j=1:a_l
        for i=1:θ_l
        vl[j][:,i]=[s_amp[j]*cos(theta[i]),s_amp[j]*sin(theta[i])]
        end
    end
    dis=transpose([p5*ones(θ_l) p6*ones(θ_l)])/scale_f_l
    vlT=[dis*norm(vl[i][:,1])^2/scale_f_l+(T+reshape(ann_l([norm(vl[i][:,1]),Vel[i]-U₀],θ_[7:end-2]),2,2)/scale_f2)*(vl[i])+Array_chain([vl[i];(Vel[i]-np1)*ones(1,θ_l)],ann,pn)/scale_f for i in 1:length(Vel)]
    vlT
end

vel=15.0;a_l=5;amp=1.52
vv=[15.0,15.0+2.5/3,15.0+5/3,17.5]
i=1

bb=nf_dis(U₀,s_,vv,vv)
U_a=[1.85 1.85 1.85 1.85]
U_a=[3.0 3.0 3.0 3.0]

ind=1
b=plot_trans(vv[ind],5,U_a[ind])
axis = @pgf Axis(

    {xlabel=L"$h$ (m) ",
                ylabel =L"$\alpha$ (rad) ",
                legend_pos  = "south east",
                height="9cm",
                width="12cm","no marks",ymin=-9e-2,ymax=8e-2,xmin=-5e-2,xmax=5e-2}

)

@pgf for i in 1:5
    a = Plot(

        Coordinates(b[i][1,:]*sens,b[i][2,:])
    )
push!(axis, a)
end
axis
pgfsave("./Figures/exp/Ut_15.pdf",axis)

ind=2
b=plot_trans(vv[ind],5,U_a[ind])
axis = @pgf Axis(

    {xlabel=L"$h$ (m) ",
                ylabel =L"$\alpha$ (rad) ",
                legend_pos  = "south east",
                height="9cm",
                width="12cm","no marks",ymin=-9e-2,ymax=8e-2,xmin=-5e-2,xmax=5e-2}

)

@pgf for i in 1:5
    a = Plot(

        Coordinates(b[i][1,:]*sens,b[i][2,:])
    )
push!(axis, a)
end
axis
pgfsave("./Figures/exp/Ut_1583.pdf",axis)

ind=3
b=plot_trans(vv[ind],5,U_a[ind])
axis = @pgf Axis(

    {xlabel=L"$h$ (m) ",
                ylabel =L"$\alpha$ (rad) ",
                legend_pos  = "south east",
                height="9cm",
                width="12cm","no marks",ymin=-9e-2,ymax=8e-2,xmin=-5e-2,xmax=5e-2}

)

@pgf for i in 1:5
    a = Plot(

        Coordinates(b[i][1,:]*sens,b[i][2,:])
    )
push!(axis, a)
end
axis
pgfsave("./Figures/exp/Ut_16_66.pdf",axis)

ind=4
b=plot_trans(vv[ind],5,U_a[ind])
axis = @pgf Axis(

    {xlabel=L"$h$ (m) ",
                ylabel =L"$\alpha$ (rad) ",
                legend_pos  = "south east",
                height="9cm",
                width="12cm","no marks",ymin=-9e-2,ymax=8e-2,xmin=-5e-2,xmax=5e-2}

)

@pgf for i in 1:5
    a = Plot(

        Coordinates(b[i][1,:]*sens,b[i][2,:])
    )
push!(axis, a)
end
axis
pgfsave("./Figures/exp/Ut_17_5.pdf",axis)

[p5,p6]*norm(uu[i])^2/scale_f_l+(T+reshape(ann_l([norm(uu[i][:,1]),vel-U₀],θ_[7:end-1]),2,2)/scale_f2)*uu[i]+ann([uu[i];vel-np1],pn)/scale_f

function U_trans1(x,y) # plotting transformation
    p1,p2,p3,p4,p5,p6=θ_[1:6]
    T=[p1 p3;p2 p4]
    U₀=θ_[end-1];np2=θ_[end]
    pn=θ_n
    uu=[x,y]
    z=[p5,p6]*norm(uu)^2/scale_f_l+(T+reshape(ann_l([norm(uu),vel-U₀],θ_[7:end-2]),2,2)/scale_f2)*uu+ann([uu;vel-U₀],pn)/scale_f
    z[1]
end

function U_trans2(x,y) # plotting transformation
    p1,p2,p3,p4,p5,p6=θ_[1:6]
    T=[p1 p3;p2 p4]
    U₀=θ_[end-1];np2=θ_[end]
    pn=θ_n
    uu=[x,y]
    z=[p5,p6]*norm(uu)^2/scale_f_l+(T+reshape(ann_l([norm(uu),vel-U₀],θ_[7:end-2]),2,2)/scale_f2)*uu+ann([uu;vel-U₀],pn)/scale_f
    z[2]
end

x = range(-3; stop = 3, length = 50)
y = range(-3; stop = 3, length = 50)
t=range(0,2π,length=100)

##Transformation at wind speed 15

vel=15
s_amp=sqrt(s_/2+sqrt(s_^2+4*(vel-U₀))/2)
u_amp=sqrt(s_/2-sqrt(s_^2+4*(vel-U₀))/2)

axis = @pgf Axis(

    {xlabel=L"$u_1$ ",
                ylabel =L"$u_2$",
                zlabel=L"$U_1(u_1,u_2,\mu-\mu_0)$",
                legend_pos  = "north east",
                height="9cm",
                width="12cm","no marks",
#                colorbar,
#        "colormap/jet"
        },
#        Legend(["Stable LCO", "Unstable LCO"])

)

x1=s_amp*cos.(t)
y1=s_amp*sin.(t)
x2=u_amp*cos.(t)
y2=u_amp*sin.(t)

b=@pgf Plot3(
    {color="blue",
    style ={dashed,thick},
        no_marks,
    },
    Coordinates(x2, y2, -0.3*ones(length(x2))+zeros(length(x2)))
)
push!(axis, b)

b=@pgf Plot3(
    {   color="red",
    style = {thick},
        no_marks,
    },
    Coordinates(x1, y1, -0.3*ones(length(x1))+zeros(length(x1)))
)
push!(axis, b)

#add stable LCO
a=@pgf Plot3(
    {
        surf,
    },
    Coordinates(x, y, U_trans1.(x, y'))
)
push!(axis, a)

b=@pgf Plot3(
    { color="red",
    style = {thick},
        no_marks,
    },
    Coordinates(x1, y1, U_trans1.(x1, y1))
)
push!(axis, b)

b=@pgf Plot3(
    {color="blue",
    style ={dashed,thick},
        no_marks,
    },
    Coordinates(x2, y2, U_trans1.(x2, y2))
)
push!(axis, b)

pgfsave("./Figures/exp/U1surf_15.pdf",axis)

axis = @pgf Axis(

    {xlabel=L"$u_1$ ",
                ylabel =L"$u_2$",
                zlabel=L"$U_2(u_1,u_2,\mu-\mu_0)$",
                legend_pos  = "north east",
                height="9cm",
                width="12cm","no marks",
#                colorbar,
#        "colormap/jet"
        },
#        Legend(["Stable LCO", "Unstable LCO"])

)

x1=s_amp*cos.(t)
y1=s_amp*sin.(t)
x2=u_amp*cos.(t)
y2=u_amp*sin.(t)

b=@pgf Plot3(
    {color="blue",
    style ={dashed,thick},
        no_marks,
    },
    Coordinates(x2, y2, -0.3*ones(length(x2))+zeros(length(x2)))
)
push!(axis, b)

b=@pgf Plot3(
    {   color="red",
    style = {thick},
        no_marks,
    },
    Coordinates(x1, y1, -0.3*ones(length(x1))+zeros(length(x1)))
)
push!(axis, b)

#add stable LCO
a=@pgf Plot3(
    {
        surf,
    },
    Coordinates(x, y, U_trans2.(x, y'))
)
push!(axis, a)

b=@pgf Plot3(
    { color="red",
    style = {thick},
        no_marks,
    },
    Coordinates(x1, y1, U_trans2.(x1, y1))
)
push!(axis, b)

b=@pgf Plot3(
    {color="blue",
    style ={dashed,thick},
        no_marks,
    },
    Coordinates(x2, y2, U_trans2.(x2, y2))
)
push!(axis, b)
pgfsave("./Figures/exp/U2surf_15.pdf",axis)
## Transformation at wind speed 17.5
vel=17.5
s_amp=sqrt(s_/2+sqrt(s_^2+4*(vel-U₀))/2)
u_amp=sqrt(s_/2-sqrt(s_^2+4*(vel-U₀))/2)
x1=s_amp*cos.(t)
y1=s_amp*sin.(t)
x2=u_amp*cos.(t)
y2=u_amp*sin.(t)

axis = @pgf Axis(

    {xlabel=L"$u_1$ ",
                ylabel =L"$u_2$",
                zlabel=L"$U_1(u_1,u_2,\mu-\mu_0)$",
                legend_pos  = "north east",
                height="9cm",
                width="12cm","no marks",
#                colorbar,
#        "colormap/jet"
        },
#        Legend(["Stable LCO", "Unstable LCO"])

)

b=@pgf Plot3(
    {color="blue",
    style ={dashed,thick},
        no_marks,
    },
    Coordinates(x2, y2, -0.4*ones(length(x2))+zeros(length(x2)))
)
push!(axis, b)

b=@pgf Plot3(
    {   color="red",
    style = {thick},
        no_marks,
    },
    Coordinates(x1, y1, -0.4*ones(length(x1))+zeros(length(x1)))
)
push!(axis, b)

#add stable LCO
a=@pgf Plot3(
    {
        surf,
    },
    Coordinates(x, y, U_trans1.(x, y'))
)
push!(axis, a)

b=@pgf Plot3(
    { color="red",
    style = {thick},
        no_marks,
    },
    Coordinates(x1, y1, U_trans1.(x1, y1))
)
push!(axis, b)

b=@pgf Plot3(
    {color="blue",
    style ={dashed,thick},
        no_marks,
    },
    Coordinates(x2, y2, U_trans1.(x2, y2))
)
push!(axis, b)

pgfsave("./Figures/exp/U1surf_175.pdf",axis)

axis = @pgf Axis(

    {xlabel=L"$u_1$ ",
                ylabel =L"$u_2$",
                zlabel=L"$U_2(u_1,u_2,\mu-\mu_0)$",
                legend_pos  = "north east",
                height="9cm",
                width="12cm","no marks",
#                colorbar,
#        "colormap/jet"
        },
#        Legend(["Stable LCO", "Unstable LCO"])

)

x1=s_amp*cos.(t)
y1=s_amp*sin.(t)
x2=u_amp*cos.(t)
y2=u_amp*sin.(t)

b=@pgf Plot3(
    {color="blue",
    style ={dashed,thick},
        no_marks,
    },
    Coordinates(x2, y2, -0.3*ones(length(x2))+zeros(length(x2)))
)
push!(axis, b)

b=@pgf Plot3(
    {   color="red",
    style = {thick},
        no_marks,
    },
    Coordinates(x1, y1, -0.3*ones(length(x1))+zeros(length(x1)))
)
push!(axis, b)

#add stable LCO
a=@pgf Plot3(
    {
        surf,
    },
    Coordinates(x, y, U_trans2.(x, y'))
)
push!(axis, a)

b=@pgf Plot3(
    { color="red",
    style = {thick},
        no_marks,
    },
    Coordinates(x1, y1, U_trans2.(x1, y1))
)
push!(axis, b)

b=@pgf Plot3(
    {color="blue",
    style ={dashed,thick},
        no_marks,
    },
    Coordinates(x2, y2, U_trans2.(x2, y2))
)
push!(axis, b)
pgfsave("./Figures/exp/U2surf_175.pdf",axis)

@save "./saved_file/ML_flutter_exp.jld" p θ_ θ_n θ t_series t_series2 # save the results
@load "./saved_file/ML_flutter_exp.jld" p θ_ θ_n θ t_series t_series2
θ_t=vcat(θ_,θ_n)
ll=length(θ_);U₀=θ_[end-1];s_=θ_[end];sens=18/100
