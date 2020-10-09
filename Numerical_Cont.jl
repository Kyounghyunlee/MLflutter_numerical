using DifferentialEquations, NLsolve, Plots, Statistics, LinearAlgebra, StaticArrays
using ForwardDiff
#=
Numerical_Cont.jl is able to do numerical continuation of system with Hopf bifurcation starting from the stable LCO solutions.
Equation of motion should be expressed in nonmutating form as flutter_eq_CBC_nonmu(u, p, t)
and the dimension of the equation should be defined as: dimension(::typeof(flutter_eq_CBC_nonmu)) = 6
=#

function flutter_eq_CBC_nonmu(u, p, t) #Flutter equation of motion for Model 1 (nonmutating form)
    μ=p[1]
    Kp=p[2]
    Kd=p[3]
    c=p[4:end]
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
    du=J*u
    #Control added on heave
    M2=inv(MM)
    control=Kp*(r*cos(theta)-u[ind1])+Kd*(r*sin(theta)-u[ind2])
    ka2=751.6; ka3=5006;
    nonlinear_h=-M2[1,2]*(ka2*u[3]^2+ka3*u[3]^3)
    nonlinear_theta=-M2[2,2]*(ka2*u[3]^2+ka3*u[3]^3)

    du[2]+=nonlinear_h+M2[1,1]*control
    du[4]+=nonlinear_theta+M2[2,1]*control
    return du
end

function flutter_eq_nonmu(u, p, t) #Flutter equation of motion for Model 1 (nonmutating form)
    μ=p[1]
    ind1=1;ind2=3;
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
    du=J*u
    #Control added on heave
    M2=inv(MM)
    ka2=751.6; ka3=5006;
    nonlinear_h=-M2[1,2]*(ka2*u[3]^2+ka3*u[3]^3)
    nonlinear_theta=-M2[2,2]*(ka2*u[3]^2+ka3*u[3]^3)

    du[2]+=nonlinear_h
    du[4]+=nonlinear_theta
    return du
end

dimension(::typeof(flutter_eq_CBC_nonmu)) = 6

function fourier_diff(N::Integer)
    # For a MATLAB equivalent see http://appliedmaths.sun.ac.za/~weideman/research/differ.html
    h = 2π/N
    D = zeros(N, N)
    # First column
    n = ceil(Int, (N-1)/2)
    if (N % 2) == 0
        for i in 1:n
            D[i+1, 1] = -((i % 2) - 0.5)*cot(i*h/2)
        end
    else
        for i in 1:n
            D[i+1, 1] = -((i % 2) - 0.5)*csc(i*h/2)
        end
    end
    for i in n+2:N
        D[i, 1] = -D[N-i+2, 1]
    end
    # Other columns (circulant matrix)
    for j in 2:N
        D[1, j] = D[N, j-1]
        for i in 2:N
            D[i, j] = D[i-1, j-1]
        end
    end
    return D
end


function periodic_zero_problem(rhs, u, p, ee)
    # Assume that u has dimensions nN + 1 where n is the dimension of the ODE
    n = dimension(rhs)
    N = (length(u)-1)÷n
    T = u[end]/(2π)  # the differentiation matrix is defined on [0, 2π]
    # Evaluate the right-hand side of the ODE
    res=Array{Float64}(undef,n*N+1,1)
    for i in 1:n:n*N
        res[i:i+n-1] = T*rhs(@view(u[i:i+n-1]), p, 0)  # use a view for speed
    end
    D=fourier_diff(N)
    # Evaluate the derivative; for-loop for speed equivalent to u*Dᵀ
    for (i, ii) in pairs(1:n:n*N)
        # Evaluate the derivative at the i-th grid point
        for (j, jj) in pairs(1:n:n*N)
            for k = 1:n
                res[ii+k-1] -= D[i, j]*u[jj+k-1]
            end
        end
    end
    res[end] = (u[1] - ee)
    return res
end

function periodic_zero_problem2(rhs, u, p, ee, dv, pu, ds) # Including Pseudo arc-length equation to periodic zero problem
    n = dimension(rhs)
    N = (length(u)-1)÷n
    res=Array{Float64}(undef,n*N+2,1)
    res1=periodic_zero_problem(rhs, u, p, ee)
    p1=p[1];
    uu=[u;p1]
    du=uu-pu

    res[1:end-1]=res1
    arclength=norm(transpose(dv)*du)
    res[end] = arclength-ds  #Pseudo-arclength equation
    return res
end

function periodic_zero_J(jeq, em, u, p, ee) # Jacobian matirx of the periodic zero problem
    n = dimension(em)
    N = (length(u)-1)÷n
    T = u[end]/(2π) # the differentiation matrix is defined on [0, 2π]
    J = zeros(n*N+1,n*N+1)
    D=fourier_diff(N)
    # Evaluate the right-hand side of the ODE
    for i in 1:n:n*N
        J[i:i+n-1,i:i+n-1] = T*jeq(@view(u[i:i+n-1]), p, 0)  # use a view for speed
        J[i:i+n-1,n*N+1] = em(@view(u[i:i+n-1]), p, 0)
    end
    for (i, ii) in pairs(1:n:n*N)
        for (j, jj) in pairs(1:n:n*N)
            for k = 1:n
                J[ii+k-1,jj+k-1] -= D[i, j]
            end
        end
    end
    J[n*N+1,1:n*N+1] = zeros(1,n*N+1)
    J[n*N+1,1] = 1
    return J
end

function periodic_zero_J2(jeq, jeq2, em, u, p, ee, dv) # Jacobian matirx of the periodic zero problem 2
    n=dimension(em)
    N = (length(u)-1)÷n
    J = zeros(n*N+2,n*N+2)
    T = u[end]/(2π)
    J[1:n*N+1,1:n*N+1]=periodic_zero_J(jeq, em, u, p, ee)
    for i in 1:n:n*N
        J[i:i+n-1,n*N+2] = T*jeq2(@view(u[i:i+n-1]), p, 0)
    end
    J[n*N+2,:]=dv
    return J
end

function LCO_solve(u1,ini_T,eq,jeq,p,ee,z_tol) # Solving periodic zero problem using Newton methods
    u0=[u1;ini_T]
    err=0
    for i in 1:50
        res=periodic_zero_problem(eq, u0, p, ee)
        u1=u0-periodic_zero_J(jeq, eq, u0, p, ee)\res
        u0=u1
        er=transpose(res)*res
        err=vcat(err,er[1,1])
        if er[1,1]<z_tol
            break
        end
    end
    return (u=u0, err=err)
end

function LCO_solve2(u1,ini_T,eq,jeq,jeq2,p,ee,z_tol,dv,pu,ds) # Solving periodic zero problem 2 using Newton methods
    p0=p[1]
    u0=[u1;ini_T]
    uu0=[u0;p0]
    err=zeros(0,1)
    np=p
    for i in 1:50
        res=periodic_zero_problem2(eq, u0, np, ee, dv, pu, ds)
        uu1=uu0-periodic_zero_J2(jeq, jeq2, eq, u0, np, ee, dv)\res
        uu0=uu1
        u0=uu0[1:end-1]
        p0=uu0[end]
        np=[p0 transpose(p[2:end])]
        er=norm(transpose(res)*res)
        err=vcat(err,er)
        if er<z_tol
            break
        end
    end
    return (u=uu0, err=err)
end

function zero_measure(u,ind,t) # Numerical measure of the zero crossing point of the time series
    # ind is the index number of the monitored signal
    l=length(u)
    zero=Array{Float64}(undef, 0)
    T=Array{Float64}(undef, 0)
    low_p=Array{Float64}(undef, 0)
    high_p=Array{Float64}(undef, 0)
    Ti=Array{Int64}(undef, 0)
    for i in 2:l-1
        sign_con2=u[i][ind+1]*u[i-1][ind+1]
        if sign_con2 < 0
            if (u[i][ind]+u[i-1][ind])/2 < 0
                low_p=vcat(low_p,(u[i][ind]+u[i-1][ind])/2)
            else
                high_p=vcat(high_p,(u[i][ind]+u[i-1][ind])/2)
            end
        end
    end
    h₀=mean(high_p)+mean(low_p)
    h₀=h₀/2
    for i in 2:l-1
        sign_con=(u[i][ind]-h₀)*(u[i+1][ind]-h₀)
        if sign_con < 0
            if (u[i][ind+1]+u[i+1][ind+1])/2 > 0
                zero=vcat(zero,(u[i][ind]+u[i-1][ind])/2)
                Ti=vcat(Ti,i)
                T=vcat(T,t[i])
            end
        end
    end
    t_l=length(T)
    P=Array{Float64}(undef, t_l-1)
    for j in 2:t_l
        P[j-1]=T[j]-T[j-1]
    end
    return (T=Ti, P=P, hp=high_p, lp=low_p, h₀=h₀)
end

function get_sol(u0,N,ind1,ind2) # convert solution of collocation -> Array form
    # returns phase angle, amplitude
    dim=Int((length(u0)-1)/N)
    u=u0[1:end-1]
    T=u0[end]
    uu=Array{Float64}(undef,dim,N)
    theta=Array{Float64}(undef,N)
    r=Array{Float64}(undef,N)
    for i in 1:dim
        uu[i,:]=u[i:dim:end]
    end
    for i in 1:N
        theta[i]=atan(uu[ind2,i],uu[ind1,i])
        r[i]=sqrt(uu[ind1,i]^2+uu[ind2,i]^2)
    end
    return (u=uu,T=T,t=theta,r=r)
end

function get_sol2(sol,ind) # Convert sol of differentiation package to time series of u[ind]
    lu=length(sol.u)
    u=Vector{Float64}(undef, lu)
    for i in 1:lu
        uv=sol.u[i]
        u[i]=uv[ind]
    end
    return u
end

function get_sol_polar(c,tl,u₀,v₀) # Convert solution polar -> Cartesian
    r=c[1]*ones(tl)
    nh=Int((length(c)-1)/2)
    θ=range(0, stop = 2π, length = tl)
    for i in 1:nh
        for j in 1:tl
            r[j]+=c[i+1]*cos(θ[j]*i)
            r[j]+=c[i+nh+1]*sin(θ[j]*i)
        end
    end
    u=u₀*ones(tl)+r.*cos.(θ)
    v=v₀*ones(tl)+r.*sin.(θ)
    return (u=u,v=v)
end

function get_stable_LCO(p,u0,tl,tol,eq,stol,rp,ind1,ind2,u₀,v₀) # Get a stable LCO from numerical integration
    u=u0
    dim=length(u0)
    prob = ODEProblem(eq,u,(0,tl*rp),p)
    sol = DifferentialEquations.solve(prob,Tsit5(),reltol=stol,abstol=stol)
    vP=1
    P=0
    T=0
    while vP>tol
        u=sol.u[end]
        prob = ODEProblem(eq,u,(0,tl),p)
        sol = DifferentialEquations.solve(prob,Tsit5(),reltol=stol,abstol=stol)
        z=zero_measure(sol.u,1,sol.t)
        vP=Statistics.var(z.hp)
        P=z.P[1]
        T=z.T
    end
    tl=length(sol)
    uu=Array{Float64}(undef,tl,dim)
    for i in 1:dim
        uu[:,i]=get_sol2(sol,i)
    end
    t=Array{Float64}(undef,length(sol))
    r=Array{Float64}(undef,length(sol))
    u=uu[:,ind1];v=uu[:,ind2]
    for i in 1:length(u)
        t[i]=atan(v[i]-v₀,u[i]-u₀)
        r[i]=sqrt((u[i]-u₀)^2+(v[i]-v₀)^2)
    end
    return (u=uu,t=t,r=r,P=P,T=T)
end

function get_stable_LCO(p,u0,tl,tol,eq,stol,rp,ind1,ind2,u₀,v₀,st) # Get a stable LCO from numerical integration
    u=u0
    dim=length(u0)
    prob = ODEProblem(eq,u,(0,tl*rp),p)
    sol = DifferentialEquations.solve(prob,Tsit5(),reltol=stol,abstol=stol,saveat=st)
    vP=1
    P=0
    T=0
    while vP>tol
        u=sol.u[end]
        prob = ODEProblem(eq,u,(0,tl),p)
        sol = DifferentialEquations.solve(prob,Tsit5(),reltol=stol,abstol=stol,saveat=st)
        z=zero_measure(sol.u,1,sol.t)
        vP=Statistics.var(z.hp)
        P=z.P[1]
        T=z.T
    end
    tl=length(sol)
    uu=Array{Float64}(undef,tl,dim)
    for i in 1:dim
        uu[:,i]=get_sol2(sol,i)
    end
    t=Array{Float64}(undef,length(sol))
    r=Array{Float64}(undef,length(sol))
    u=uu[:,ind1];v=uu[:,ind2]
    for i in 1:length(u)
        t[i]=atan(v[i]-v₀,u[i]-u₀)
        r[i]=sqrt((u[i]-u₀)^2+(v[i]-v₀)^2)
    end
    return (u=uu,t=t,r=r,P=P,T=T)
end

function MonoD(eq,u,p,t) # Variational equation for monodromy matrix computation (ForwardDiff is used for jacobian computation)
    jeq=(u,p,t) -> ForwardDiff.jacobian(u -> eq(u,p,t), u)
    n=dimension(eq)
    v=u[1:Int(n*n)]
    uu=u[Int(n*n+1):end]
    M=reshape(v, n, n)
    J=jeq(uu,p,t)
    dM=J*M
    dv=vec(dM)
    duu=eq(uu,p,t)
    duu=vec(duu)
    du=[dv;duu]
    return du
end

function Monodromy_compute(eq,u,p,N,ind) # Monodromy matrix computation using numerical integration
    n=dimension(eq)
    eye=1.0*Matrix(I, n, n)
    v0=vec(eye)
    g=get_sol(u,N,ind[1],ind[2])
    T=g.T
    uu=g.u
    M=eye
    tl2=T/N
    for i in 1:N
        uu0=uu[:,i]
        u0=[v0;uu0]
        prob = ODEProblem((u,p,t) -> MonoD(eq,u,p,t),u0,(0,tl2),p)
        sol = solve(prob,Tsit5(),reltol=1e-11,abstol=1e-11)
        w=sol.u[end]
        m1=w[1:Int(n*n)]
        M1=reshape(m1, n, n)
        M=M1*M
    end
    Eig1=eigen(M)
    μ=Eig1.values
    return μ
end

function LS_harmonics(r,t,ω,N) # Computing least square fit of the fourier coefficients of the amplitude in the projected phase plane
    c=Array{Float64}(undef,2*N+1)
    M=Array{Float64}(undef,1,2*N+1)
    tM=Array{Float64}(undef,0,2*N+1)
    tl=length(t)
    rr=Array{Float64}(undef,tl)
    M[1]=1
    for j in 1:tl
        for i in 1:N
            M[1+i]=cos(ω*t[j]*i)
            M[1+N+i]=sin(ω*t[j]*i)
        end
        tM=vcat(tM,M)
    end
    MM=transpose(tM)*tM
    rN=transpose(tM)*r
#    MM=SMatrix{2*N+1,2*N+1}(MM)
#    rN=SVector{2*N+1}(rN)
    c=inv(MM)*rN
    for j in 1:tl
        rr[j]=c[1]
        for i in 1:N
            rr[j]+=c[i+1]*cos(ω*t[j]*i)
            rr[j]+=c[i+1+N]*sin(ω*t[j]*i)
        end
    end
    return (coeff=c,rr=rr)
end

function ini_val(p,tl,tol,N,eq,u0,stol,rp,ind) #Get stable LCO from the ODE numerical solutions- initial collocation points
    dim=length(u0)
    s0=get_stable_LCO(p,u0,tl,tol,eq,stol,rp,ind[1],ind[2],0,0)
    pp=s0.P[1]
    ini_T=pp
    mu=s0.u[:,1]
    u=mu-1e-3*ones(length(mu))
    u=broadcast(abs, u)
    ps=findall(isequal(minimum(u)), u)
    ps=ps[1]
    u0= s0.u[ps,:]
    #Get a time series of periodic solution
    tl2=pp/N
    ini_u=Array{Float64}(undef,Int(dim*N))
    ini_u[1:dim]=u0
    #uu=pu+ds*dv;
    for i in 1:N-1
        prob = ODEProblem(eq,u0,(0,tl2),p)
        sol = DifferentialEquations.solve(prob,Tsit5(),reltol=1e-11,abstol=1e-11)
        u0=sol.u[end]
        ini_u[Int(dim*i+1):Int(dim*i+dim)]=u0
    end
    return (u=ini_u,T=ini_T)
end

function continuation_Hopf_SLCO(eq,tol,N,sp,p0,ds,u0,tl,ind) #Start numerical continuation from the stable LCO
    jeq=(u,p,t) -> ForwardDiff.jacobian(u -> eq(u,p,t), u)
    jeq2=(u,p,t) -> ForwardDiff.jacobian(p -> eq(u,p,t), p)[:,1]
    par_purt=1e-3
    ee=1e-3;
    p=p0
    p=vcat(p[1]+par_purt,p[2:end])
    stol=1e-8;rp=6;ptol=1e-3;
    s1=ini_val(p,tl,ptol,N,eq,u0,stol,rp,ind)
    s=LCO_solve(s1.u,s1.T,eq,jeq,p,ee,tol)

    p=p0
    s2=LCO_solve(s1.u,s1.T,eq,jeq,p,ee,tol)

    du=s2.u-s.u
    du=[du;-par_purt]
    dv=du/norm(du)
    dim=dimension(eq);ll=dim*N+1;
    V = [zeros(ll) for _ in 1:sp]
    P = zeros(sp,1)
    sU=p0[1]
    pu=s2.u;pu=[pu;sU]
    V[1]=vec(s2.u);P[1]=sU

    uu=pu+ds*dv
    p=vcat(uu[end],p[2:end])
    u1=uu[1:end-2];ini_T=uu[end-1];

    for i in 2:sp
        s2=LCO_solve2(u1,ini_T,eq,jeq,jeq2,p,ee,tol,dv,pu,ds)
        du=s2.u-pu;dv=du/norm(du)
        V[i]=s2.u[1:end-1];P[i]=s2.u[end]
        pu=s2.u;
        uu=pu+ds*dv;
        p=vcat(uu[end],p[2:end])
        u1=uu[1:end-2];ini_T=uu[end-1];
    end
    return (V=V,P=P)
end

function amp_LCO(U,N,ind_d,ind) # Compute amplitude of the LCO (amplitude=maximum(u[ind])-minimum(u[ind]))
    l1=length(U)
    amp=Vector{Float64}(undef, l1)
    for i in 1:l1
        g=get_sol(U[i],N,ind[1],ind[2])
        uu=g.u[ind_d,:]
        amp[i]=maximum(uu)-minimum(uu)
    end
    return amp
end


function Hopf_point(eq,u,p) # Compute amplitude of the LCO (amplitude=maximum(u[ind])-minimum(u[ind]))
jeq=(u,p,t) -> ForwardDiff.jacobian(u -> eq(u,p,t), u)
s=nlsolve(p0->maximum(real(eigen(jeq(u,p0,t)).values))-1e-8,p)
    return s.zero[1]
end
