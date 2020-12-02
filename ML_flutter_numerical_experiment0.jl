# Numercal experiment of Flutter model ML modelling
# Training data sets are generated from the ODE solutions

using OrdinaryDiffEq,ModelingToolkit,DataDrivenDiffEq,LinearAlgebra, DiffEqSensitivity, Optim,DiffEqFlux, Flux, Printf,PGFPlotsX,LaTeXStrings, JLD2, MAT
include("Numerical_Cont_Hopf_CBC.jl")
# Numerical continuation of the experimental model
#eq=flutter_eq_CBC_nonmu;
tol=1e-8; #Zero tolerance
N=100; # collocation points for continuation
sp=800; # Number of continuation points
sU=18.0; # Starting point of continuation (wind velocity)
ds=0.02; # Arclength (note that this remains constant in Numerical_Cont.jl)
#p0=[sU,0,0,0,0,0,0,0,0,0]; # Initial parameter of the eq
#u0=[0.1;0.1;0.1;0;0;0]; # initial perturbation of the system to reach stable LCO
#tl=3.0; # length of time to detect stable LCO from time integration
st=1e-3
ind=[1,3] #index of projecting plane (Not important if we do not use CBC)

@time slco=continuation_flutter(tol,N,sp,sU,ds)
U=slco.V
P=slco.P
amp=amp_LCO(U,1)
plot(P,amp)
θ_l=100

function flutter_eq_CBC(du,u, p, t) #Flutter equation of motion
    μ,Kp,Kd,Fa,Fω=p[1:5]
    c=p[6:end]
    ind1=1;ind2=3;
    theta=atan(u[ind1],u[ind2]) #theta is computed from two state variables
    r=c[1]
    h=c[2:end]
    nh=Int(length(h)/2)
    for i=1:nh
        r+=h[i]*cos(i*theta)+h[i+nh]*sin(theta*i)
    end
    b=0.15; a=-0.5; rho=1.204; x__alpha=0.2340
    c__0=1; c__1=0.1650; c__2=0.0455; c__3=0.335; c__4=0.3; c__alpha=0.562766779889303; c__h=15.4430
    I__alpha=0.1726; k__alpha=54.116182926744390; k__h=3.5294e+03; m=5.3; m__T=16.9
    U=μ

    MM = [b ^ 2 * pi * rho + m__T -a * b ^ 3 * pi * rho + b * m * x__alpha 0; -a * b ^ 3 * pi * rho + b * m * x__alpha I__alpha + pi * (0.1e1 / 0.8e1 + a ^ 2) * rho * b ^ 4 0; 0 0 1;]
    DD = [c__h + 2 * pi * rho * b * U * (c__0 - c__1 - c__3) (1 + (c__0 - c__1 - c__3) * (1 - 2 * a)) * pi * rho * b ^ 2 * U 2 * pi * rho * U ^ 2 * b * (c__1 * c__2 + c__3 * c__4); -0.2e1 * pi * (a + 0.1e1 / 0.2e1) * rho * (b ^ 2) * (c__0 - c__1 - c__3) * U c__alpha + (0.1e1 / 0.2e1 - a) * (1 - (c__0 - c__1 - c__3) * (1 + 2 * a)) * pi * rho * (b ^ 3) * U -0.2e1 * pi * rho * (U ^ 2) * (b ^ 2) * (a + 0.1e1 / 0.2e1) * (c__1 * c__2 + c__3 * c__4); -1 / b a - 0.1e1 / 0.2e1 (c__2 + c__4) * U / b;]
    KK = [k__h 2 * pi * rho * b * U ^ 2 * (c__0 - c__1 - c__3) 2 * pi * rho * U ^ 3 * c__2 * c__4 * (c__1 + c__3); 0 k__alpha - 0.2e1 * pi * (a + 0.1e1 / 0.2e1) * rho * (c__0 - c__1 - c__3) * (b ^ 2) * (U ^ 2) -0.2e1 * pi * rho * b * (U ^ 3) * (a + 0.1e1 / 0.2e1) * c__2 * c__4 * (c__1 + c__3); 0 -U / b c__2 * c__4 * U ^ 2 / b ^ 2;]

    K1=-inv(MM)*KK;
    D1=-inv(MM)*DD;

    J1=[0 1 0 0 0 0]
    J2=[K1[1,1] D1[1,1] K1[1,2] D1[1,2] K1[1,3] D1[1,3]]
    J3=[0 0 0 1 0 0]
    J4=[K1[2,1] D1[2,1] K1[2,2] D1[2,2] K1[2,3] D1[2,3]]
    J5=[0 0 0 0 0 1]
    J6=[K1[3,1] D1[3,1] K1[3,2] D1[3,2] K1[3,3] D1[3,3]]

    J=[J1;J2;J3;J4;J5;J6]
    du_=J*u
    #Control added on heave
    M2=inv(MM)
    control=Kp*(r*cos(theta)-u[ind1])+Kd*(r*sin(theta)-u[ind2])
    ka2=751.6; ka3=5006;
    nonlinear_h=-M2[1,2]*(ka2*u[3]^2+ka3*u[3]^3)
    nonlinear_theta=-M2[2,2]*(ka2*u[3]^2+ka3*u[3]^3)

    du[1]=du_[1]
    du[2]=du_[2]+nonlinear_h+M2[1,1]*control+M2[1,1]*Fa*sin(Fω*t)
    du[3]=du_[3]
    du[4]=du_[4]+nonlinear_theta+M2[2,1]*control+M2[2,1]*Fa*sin(Fω*t)
    du[5]=du_[5]
    du[6]=du_[6]
end

function generate_data(vel_l,Vel,nh) # Training data
    #Generate training data
    u0=Float32[2e-1,0,2e-1,0,0,0];
    tol=1e-7;stol=1e-8
    eq=flutter_eq_CBC;
    rp=5;ind1=1;ind2=3;
    pp=[zeros(10) for i in 1:vel_l]
    AA=zeros(vel_l,Int(nh*2+1))

    p_=zeros(9)
    p_=vcat(Vel[1],p_)
    g=get_stable_LCO(p_,u0,tl,tol,eq,stol,rp,ind1,ind2,0.0,0.0,st)

    u₀=mean(g.u[:,1]);v₀=mean(g.u[:,3])
    for i in 1:vel_l
        pp[i][1]=Vel[i]
        g=get_stable_LCO(pp[i],u0,tl,tol,eq,stol,rp,ind1,ind2,u₀,v₀,st)
        r=g.r;t=g.t;
        c=LS_harmonics(r,t,1,nh).coeff
        AA[i,:]=c
    end
    cc=[LS_harmonics(get_sol(U[s_ind[i]],N,1,3).r,get_sol(U[s_ind[i]],N,1,3).t,1,nh).coeff for i in 1:length(s_ind)]
    cc=hcat(cc)

    AA=transpose(AA)
    Al=AA
    AA=hcat(AA,cc)
    t_series=[Transpose(get_stable_LCO([Vel[i] transpose(zeros(9))],u0,tl,tol,eq,stol,rp,ind1,ind2,u₀,v₀,st).u[:,[1,3]]) for i in 1:vel_l]
    θ_series=[get_stable_LCO([Vel[i] transpose(zeros(9))],u0,tl,tol,eq,stol,rp,ind1,ind2,u₀,v₀,st).t for i in 1:vel_l]
    θ=range(0, stop = 2π, length = θ_l)
    coθ=cos.(θ)
    siθ=sin.(θ)
    return (data=AA,ts=t_series,d0=[u₀,v₀],data2=Al,coθ=coθ,siθ=siθ,theta_s=θ_series)
end

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
    np1=npv[1];np2=npv[2]
    nf=nf_dis(np1,np2,Vel,Vel2)
    vl=nf.v;vl2=nf.v2
    p1,p2,p3,p4=θ_t[1:6]
    T=[p1 p3;p2 p4]

    pl=θ_t[5:end-sl-1]
    ps=θ_t[end-sl:end]

    shift=[Array_chain(norm(vl[i][:,1])*ones(1,θ_l),ann_s,ps)/scale_f1 for i in 1:length(Vel)]

    vlT=[shift[i]+(T+reshape(ann_l([norm(vl[i][:,1]),Vel[i]-np1],pl),2,2)/scale_f2)*(vl[i]) for i in 1:length(Vel)]
    vlT2=[shift[i]+(T+reshape(ann_l([norm(vl2[i][:,1]),Vel2[i]-np1],pl),2,2)/scale_f2)*(vl2[i]) for i in 1:length(Vel2)]

    Pr=f_coeff(vlT,Vel,0,0)
    Pr2=f_coeff(vlT2,Vel2,0,0)
    PP=hcat(Pr,Pr2)
    PP
end

function predict_lt(θ_t) #predict the linear transformation
    nf=nf_dis(U₀,3.65,Vel,Vel2)
    vl=nf.v;vl2=nf.v2
    p1,p2,p3,p4,p5,p6,p7,p8=θ_t[1:8]
    T=[p1 p3;p2 p4]
    dis=transpose([p5*ones(θ_l) p6*ones(θ_l)])/(scale_f_l*1000)
    dis2=transpose([p7*ones(θ_l) p8*ones(θ_l)])/scale_f_l
    vlT=[dis*norm(vl[i][:,1])+dis2*norm(vl[i][:,1])^2+T*(vl[i]) for i in 1:length(Vel)]
    vlT2=[dis*norm(vl2[i][:,1])+dis2*norm(vl2[i][:,1])^2+T*(vl2[i]) for i in 1:length(Vel2)]

    Pr=f_coeff(vlT,Vel,u₀,v₀)
    Pr2=f_coeff(vlT2,Vel2,0,0)
    hcat(Pr,Pr2)
end

function lt_pp(θ_t) # This function gives phase portrait of the transformed system from the normal form
    np1=npv[1];np2=npv[2]
    nf=nf_dis(np1,np2,Vel,Vel2)
    vl=nf.v;vl2=nf.v2
    p1,p2,p3,p4,p5,p6,p7,p8=θ_t[1:8]
    T=[p1 p3;p2 p4]
    dis=transpose([p5*ones(θ_l) p6*ones(θ_l)])/(scale_f_l*1000)
    dis2=transpose([p7*ones(θ_l) p8*ones(θ_l)])/scale_f_l
    vlT=[dis*norm(vl[i][:,1])+dis2*norm(vl[i][:,1])^2+T*(vl[i]) for i in 1:length(Vel)]
    vlT2=[dis*norm(vl2[i][:,1])+dis2*norm(vl2[i][:,1])^2+T*(vl2[i]) for i in 1:length(Vel2)]

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
## Generating training data (Data of trajectory)

vel_l=10
Vel=range(15.0, stop = 18.0, length = vel_l)
nh=10;
s_ind=[450,500]
s_ind=[600,700]

Vel2=[P[s_ind[i]] for i in 1:length(s_ind)]
U₀=18.27 #Estimated flutter speed
θ_l=100;
dat=generate_data(vel_l,Vel,nh)
AA=dat.data
Al=dat.data2
tl2=1.0
tt=Int(tl2/st+1)
t_series=[dat.ts[i][:,1:tt] for i in 1:length(Vel)]
θ_series=[dat.theta_s[i][1:tt] for i in 1:length(Vel)]
u₀=dat.d0[1]
v₀=dat.d0[2]
siθ=dat.siθ;coθ=dat.coθ
## Generate initial guess of the parameters (Simple linear transformation with rotation)
rot=-π*0.1
R=[cos(rot) -sin(rot);sin(rot) cos(rot)]
θ=vec(1e-2*R*[2.0 0.0;0.0 3])
θ=vcat(θ,zeros(2))
npv=[18.27,3.65]

# optimization works for scale_f_l>=50 for small scale_f_l optimization does not work.

hidden=5
ann_l = FastChain(FastDense(2, hidden, tanh),FastDense(hidden, hidden, tanh), FastDense(hidden,  4))
θl = initial_params(ann_l)
θ=vcat(θ,θl)
#θ=vcat(θ,npv)
scale_f2=1e9


hidden=3
ann_s = FastChain(FastDense(1, hidden, tanh),FastDense(hidden, hidden, tanh), FastDense(hidden,  2))
θs = initial_params(ann_s)
scale_f1=5e3
sl=length(θs)
θ=vcat(θ,θs)
loss_lt(θ)

rot=-π*0.1
R=[cos(rot) -sin(rot);sin(rot) cos(rot)]
θ=vec(1e-2*R*[2.0 0.0;0.0 3])
θ=vcat(θ,zeros(4))
scale_f_l=1e2 # optimization works for scale_f_l>=50 for small scale_f_l optimization does not work.
loss_lt(θ)


res1 = DiffEqFlux.sciml_train(loss_lt, θ, ADAM(0.001), maxiters = 800)
res1 = DiffEqFlux.sciml_train(loss_lt, res1.minimizer, BFGS(initial_stepnorm=1e-3), maxiters = 10000)

res1.minimum
θ_=res1.minimizer
Ap=lt_pp(θ_)

a=@pgf Axis( {xlabel=L"$h$ (m)",
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
    LegendEntry("Learnt model"),
    Plot(
        { color="blue",
            no_marks,
        },
        Coordinates(t_series[1][1,:],t_series[1][2,:])
    ),
    LegendEntry("Underlying model")
)

pgfsave("./Figures/num_flutter/LTU15_5.pdf",a)


## Add neural network to transformation to improve the model
function predict_nt(θ_t)
    p1,p2,p3,p4,p5,p6=θ_[1:6]
    T=[p1 p3;p2 p4]
    dis=transpose([p5*ones(θ_l) p6*ones(θ_l)])/scale_f_l
    pl=θ_[5:end]
    np1,np2=θ_t[1:2]

    pn=θ_t[3:end]
    T=[p1 p3;p2 p4]

    nf=nf_dis(np1,np2,Vel,Vel2)
    vl=nf.v;vl2=nf.v2

    vlT=[dis*norm(vl[i][:,1])^2+(T+reshape(ann_l([norm(vl[i][:,1]),Vel[i]-np1],pl),2,2)/scale_f2)*(vl[i])+Array_chain([vl[i];(Vel[i]-np1)*ones(1,θ_l)],ann,pn)/scale_f for i in 1:length(Vel)]
    vlT2=[dis*norm(vl2[i][:,1])^2+(T+reshape(ann_l([norm(vl2[i][:,1]),Vel2[i]-np1],pl),2,2)/scale_f2)*(vl2[i])+Array_chain([vl2[i];(Vel2[i]-np1)*ones(1,θ_l)],ann,pn)/scale_f for i in 1:length(Vel2)]

    Pr=f_coeff(vlT,Vel,u₀,v₀)
    Pr2=f_coeff(vlT2,Vel2,0,0)
    hcat(Pr,Pr2)
end

function lt_pp_n(θ_t) # This function gives phase portrait of the transformed system from the normal form (stable LCO)
    p1,p2,p3,p4,p5,p6=θ_[1:6]
    T=[p1 p3;p2 p4]
    pl=θ_[5:end]
    dis=transpose([p5*ones(θ_l) p6*ones(θ_l)])/scale_f_l
    np1,np2=θ_t[1:2]
    nf=nf_dis(np1,np2,Vel,Vel2)
    vl=nf.v;vl2=nf.v2

    pn=θ_t[3:end]
    vlT=[dis*norm(vl[i][:,1])^2+(T+reshape(ann_l([norm(vl[i][:,1]),Vel[i]-np1],pl),2,2)/scale_f2)*(vl[i])+Array_chain([vl[i];(Vel[i]-np1)*ones(1,θ_l)],ann,pn)/scale_f for i in 1:length(Vel)]
    vlT
end

function lt_pp_n_u(θ_t,Vel2) # This function gives phase portrait of the transformed system from the normal form (unstable LCO)
    p1,p2,p3,p4,p5,p6=θ_[1:6]
    T=[p1 p3;p2 p4]
    dis=transpose([p5*ones(θ_l) p6*ones(θ_l)])/scale_f_l
    np1,np2=θ_t[1:2]
    pl=θ_[5:end]
    #nf=nf_dis(U₀,3.65,Vel,Vel2)
    nf=nf_dis(np1,np2,Vel,Vel2)
    vl=nf.v;vl2=nf.v2
    pn=θ_t[3:end]

    vlT=[dis*norm(vl2[i][:,1])^2+(T+reshape(ann_l([norm(vl2[i][:,1]),Vel2[i]-np1],pl),2,2)/scale_f2)*(vl2[i])+Array_chain([vl2[i];(Vel2[i]-np1)*ones(1,θ_l)],ann,pn)/scale_f for i in 1:length(Vel2)]
    vlT
end

function loss_nt(θ_t)
    pred = predict_nt(θ_t)
    sum(abs2, AA .- pred) # + 1e-5*sum(sum.(abs, params(ann)))
end

hidden=21
ann = FastChain(FastDense(3, hidden, tanh),FastDense(hidden, hidden, tanh), FastDense(hidden,  2))
θn = initial_params(ann)
scale_f=1e3

pp=[18.27,3.65]
θn=vcat(pp,θn)
loss_nt(θn)

res_n = DiffEqFlux.sciml_train(loss_nt, θn, ADAM(0.01), maxiters = 300)

res_n.minimum
θ_2=res_n.minimizer

Ap=lt_pp_n(θ_2)

ind=1
a=@pgf Axis( {xlabel=L"$h$ (m)",
            ylabel = L"$\alpha$ (rad)",
            legend_pos  = "north west",
            height="9cm",
            width="9cm", xmin=-3e-2,xmax=3e-2,ymax=6e-2,ymin=-9e-2},

    Plot(
        { color="red",
            no_marks
        },
        Coordinates(Ap[ind][1,:],Ap[ind][2,:])
    ),
    LegendEntry("Learnt model"),
    Plot(
        { color="blue",
            no_marks,
        },
        Coordinates(t_series[ind][1,:],t_series[ind][2,:])
    ),
    LegendEntry("Underlying model")
)

pgfsave("./Figures/num_flutter/NN_U15.pdf",a)

#Checking the phase portrait of the model (Unstable LCO)
Ap=lt_pp_n_u(θ_2,Vel2)
ind=1 # Near the fold
vv=Vel2[ind]
uu=get_sol(U[s_ind[ind]],N,1,3)

a=@pgf Axis( {xlabel=L"$h$ (m)",
            ylabel = L"$\alpha$ (rad)",
            legend_pos  = "north west",
            height="9cm",
            width="9cm",
            xmin=-3e-2,xmax=3e-2,ymax=6e-2,ymin=-9e-2},

    Plot(
        { color="red",
            no_marks
        },
        Coordinates(Ap[1][1,:],Ap[1][2,:])
    ),
    LegendEntry("Learnt model"),
    Plot(
        { color="blue",
            no_marks,
        },
        Coordinates(vcat(uu.u[1,:],uu.u[1,1]),vcat(uu.u[3,:],uu.u[3,1]))
    ),
    LegendEntry("Underlying model")
)

pgfsave("./Figures/num_flutter/ust_u17.pdf",a)

ind=2
vv=Vel2[ind]# Near the equilibrium
uu=get_sol(U[s_ind[ind]],50,1,3)

a=@pgf Axis( {xlabel=L"$h$ (m)",
            ylabel = L"$\alpha$ (rad)",
            legend_pos  = "north west",
            height="9cm",
            width="9cm",
            xmin=-3e-2,xmax=3e-2,ymax=6e-2,ymin=-9e-2,
            xtick=-1e-2:4e-3:1e-2},

    Plot(
        { color="red",
            no_marks
        },
        Coordinates(Ap[2][1,:],Ap[2][2,:])
    ),
    LegendEntry("Learnt model"),
    Plot(
        { color="blue",
            no_marks,
        },
        Coordinates(vcat(uu.u[1,:],uu.u[1,1]),vcat(uu.u[3,:],uu.u[3,1]))
    ),
    LegendEntry("Underlying model")
)

pgfsave("./Figures/num_flutter/ust_u179.pdf",a)

u_ind=[400,750]
Vel3=[P[u_ind[i]] for i in 1:length(u_ind)]
Ap=lt_pp_n_u(θ_2,Vel3)

ind=1
vv=Vel3[ind]
uu=get_sol(U[u_ind[ind]],N,1,3)

a=@pgf Axis( {xlabel=L"$h$ (m)",
            ylabel = L"$\alpha$ (rad)",
            legend_pos  = "north west",
            height="9cm",
            width="9cm",
            xmin=-3e-2,xmax=3e-2,ymax=6e-2,ymin=-9e-2
},

    Plot(
        { color="red",
            no_marks
        },
        Coordinates(Ap[1][1,:],Ap[1][2,:])
    ),
    LegendEntry("Learnt model"),
    Plot(
        { color="blue",
            no_marks,
        },
        Coordinates(vcat(uu.u[1,:],uu.u[1,1]),vcat(uu.u[3,:],uu.u[3,1]))
    ),
    LegendEntry("Underlying model")
)
pgfsave("./Figures/num_flutter/ust_u153.pdf",a)

ind=2 # Near the equilibrium
vv=Vel3[ind]
uu=get_sol(U[u_ind[ind]],50,1,3)

a=@pgf Axis( {xlabel=L"$h$ (m)",
            ylabel = L"$\alpha$ (rad)",
            legend_pos  = "north east",
            height="9cm",
            width="9cm",
            xmin=-3e-2,xmax=3e-2,ymax=6e-2,ymin=-9e-2
},

    Plot(
        { color="red",
            no_marks
        },
        Coordinates(Ap[2][1,:],Ap[2][2,:])
    ),
    LegendEntry("Learnt model"),
    Plot(
        { color="blue",
            no_marks,
        },
        Coordinates(vcat(uu.u[1,:],uu.u[1,1]),vcat(uu.u[3,:],uu.u[3,1]))
    ),
    LegendEntry("Underlying model")
)
pgfsave("./Figures/num_flutter/ust_u1813.pdf",a)

θ=[θ_;θ_2]
ll=length(θ_)
θ_t=θ
## Compare the bifurcation diagram
function lt_b_dia(θ_t,ind)
    vel_l=300
    p1,p2,p3,p4,p5,p6=θ_t[1:6]
    pl=θ_t[7:ll]
    T=[p1 p3;p2 p4]
    np1,np2=θ_t[ll+1:ll+2]
    Vel=range(np1-np2^2/4+1e-7, stop = np1, length = vel_l)
    nf=nf_dis(np1,np2,Vel,Vel)
    vl=nf.v;vl2=nf.v2
    pn=θ_t[ll+3:end]

    dis=transpose([p5*ones(θ_l) p6*ones(θ_l)])/scale_f_l
    vlT=[dis*norm(vl[i][:,1])^2+(T+reshape(ann_l([norm(vl[i][:,1]),Vel[i]-np1],pl),2,2)/scale_f2)*(vl[i])+Array_chain([vl[i];(Vel[i]-U₀)*ones(1,θ_l)],ann,pn)/scale_f for i in 1:length(Vel)]
    vlT2=[dis*norm(vl2[i][:,1])^2+(T+reshape(ann_l([norm(vl2[i][:,1]),Vel[i]-np1],pl),2,2)/scale_f2)*(vl2[i])+Array_chain([vl2[i];(Vel[i]-U₀)*ones(1,θ_l)],ann,pn)/scale_f for i in 1:length(Vel)]
    vlTas=[maximum(vlT[i][ind,:])-minimum(vlT[i][ind,:]) for i in 1:length(Vel)]
    vlTau=[maximum(vlT2[i][ind,:])-minimum(vlT2[i][ind,:]) for i in 1:length(Vel)]
    return (s=vlTas,u=vlTau,v=Vel)
end

bd=lt_b_dia(θ,1)
h=[maximum(t_series[i][1,:])-minimum(t_series[i][1,:]) for i in 1:length(Vel)]
d_amp=[amp[s_ind[i]] for i in 1:length(s_ind)]
d_P=[P[s_ind[i]] for i in 1:length(s_ind)]
P=vec(P)
amp=vec(amp)
vv=vcat(bd.v,bd.v)
aa=vcat(bd.s,bd.u)

#Plot bifurcation diagram
a=@pgf Axis( {xlabel="Air speed (m/sec)",
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
        { color="blue",
            no_marks,
        },
        Coordinates(vec(P),amp)
    ),
    LegendEntry("Underlying model"),
    Plot(
        { color="red",
            no_marks
        },
        Coordinates(bd.v,bd.s)
    ),
    LegendEntry("Learnt model"),

    Plot(
        { color="red",
            no_marks,
        },
        Coordinates(bd.v,bd.u)
    ),
)

pgfsave("./Figures/num_flutter/bd_flutter.pdf",a)

function Inv_T_u(th0,vel,tol) # This function gives initial conditions of the model
    p1,p2,p3,p4,p5,p6=θ_[1:6]
    T=[p1 p3;p2 p4]
    np1,np2=θ_2[1:2]
    pn=θ_2[3:end]
    s_amp=sqrt(np2/2+sqrt(np2^2+4*(vel-np1))/2)
    theta=range(-π,stop=π,length=50000)

    uu=[[s_amp*cos(theta[i]),s_amp*sin(theta[i])] for i in 1:length(theta)]
    u=[[p5,p6]*norm(uu[i])^2/scale_f_l+T*uu[i]+ann([uu[i];vel-np1],pn)/scale_f for i in 1:length(theta)]
    t0=[abs(atan(u[i][2],u[i][1])-th0) for i in 1:length(theta)]
    er=minimum(t0)
    while er>tol
    #    global theta,t0
        theta=range(theta[argmin(t0)-1],theta[argmin(t0)+1],length=500)
        uu=[[s_amp*cos(theta[i]),s_amp*sin(theta[i])] for i in 1:length(theta)]
        u=[[p5,p6]*norm(uu[i])^2/scale_f_l+T*uu[i]+ann([uu[i];vel-np1],pn)/scale_f for i in 1:length(theta)]
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
    θ=u[1]
    r=u[2]
    c=u[3]
    np1,np2=θ_2[1:2]
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
    np1,np2=θ_2[1:2]
    pn=θ_2[3:end]
    p1,p2,p3,p4,p5,p6=θ_[1:6]
    T=[p1 p3;p2 p4]
    A1=[Array(concrete_solve(ODEProblem(dudt_ph,u_t0[i],(0,tl2),p), Tsit5(), u_t0[i], p, saveat = st,
                         abstol=1e-8, reltol=1e-8,
                         sensealg = InterpolatingAdjoint(autojacvec=ReverseDiffVJP()))) for i in 1:length(Vel)]
    uu=[transpose(hcat(A1[i][2,:].*cos.(A1[i][1,:]),A1[i][2,:].*sin.(A1[i][1,:]),A1[i][3,:])) for i in 1:vl]
    delU=zeros(2,spl)
    delU2=-np1*ones(1,spl)
    delU=vcat(delU,delU2)
    uu=[uu[i]+delU for i in 1:vl]
    dis=transpose([p5*ones(spl) p6*ones(spl)])/scale_f_l

    vlT=[dis*norm(uu[i][1:2])^2+T*uu[i][1:2,:]+Array_chain(uu[i],ann,pn)/scale_f for i in 1:vl]
    Pr=zeros(0,spl)
    for i in 1:vl
        theta=vlT[i][[1,2],:]
        Pr=vcat(Pr,theta)
    end
Pr
end

tl2=1.0
spl=Int(tl2/st+1)
vl=length(Vel)
hidden=31
ann3 = FastChain(FastDense(3, hidden, tanh),FastDense(hidden, 1, tanh))
np = initial_params(ann3)
omega=15.3
p = vcat(omega,np)

# Generate data and initial θ
np1,np2=θ_2[1:2]
pn=θ_2[3:end]
s_amp=[sqrt(np2/2+sqrt(np2^2+4*(Vel[i]-np1))/2) for i in 1:length(Vel)]
u_amp=[sqrt(np2/2-sqrt(np2^2+4*(Vel2[i]-np1))/2) for i in 1:length(Vel2)]
theta0=[θ_series[i][1] for i in 1:length(Vel)]
tol=1e-5
θ₀=[Inv_T_u(theta0[i],Vel[i],tol) for i in 1:length(Vel)]
u_t0=[[θ₀[i],s_amp[i],Vel[i]] for i in 1:length(Vel)]
u_t02=[[s_amp[i]*cos.(θ₀[i]),s_amp[i]*sin.(θ₀[i]),Vel[i]] for i in 1:length(Vel)]

uu_t0=[[θ₀[i],u_amp[i],Vel2[i]] for i in 1:length(Vel2)]
uu_t02=[[u_amp[i]*cos.(θ₀[i]),u_amp[i]*sin.(θ₀[i]),Vel2[i]] for i in 1:length(Vel2)]

spl=length(t_series[1][1,:])
t_s=zeros(vl*2,spl)
for i in 1:vl
    t_s[[2*(i-1)+1,2*(i-1)+2],:]=t_series[i]
end
A3=t_s

function loss_time_T(p)
    pred = predict_time_T(p)
    sum(abs2, A3 .- pred)
end

om_scale=0.3
loss_time_T(p)
res_t = DiffEqFlux.sciml_train(loss_time_T, p, ADAM(0.01), maxiters = 100)
res_t = DiffEqFlux.sciml_train(loss_time_T, res1.minimizer, BFGS(initial_stepnorm=1e-3), maxiters = 10000)

res_t.minimum
p=res_t.minimizer

tv=range(0,1,length=1001)
a=@pgf Axis( {xlabel="Time (sec)",
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
    LegendEntry("Learnt model"),
    Plot(
        { color="blue",
            no_marks,
        },
        Coordinates(tv,t_series[1][1,:])
    ),
    LegendEntry("Underlying model")
)

pgfsave("./Figures/num_flutter/time_u15.pdf",a)

a=@pgf Axis( {xlabel="Time (sec)",
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
    LegendEntry("Learnt model"),
    Plot(
        { color="blue",
            no_marks,
        },
        Coordinates(tv,t_series[10][1,:])
    ),
    LegendEntry("Underlying model")
)
pgfsave("./Figures/num_flutter/time_u18.pdf",a)

@save "./saved_file/ML_flutter_num.jld" p θ_ θ_2 θ t_series ann ann3  #save the results
@load "./saved_file/ML_flutter_num.jld" p θ_ θ_2 θ t_series ann ann3
