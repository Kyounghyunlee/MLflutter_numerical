#Numerical experiment for the CBC experiment of the flutter model control is added at the heave
using DifferentialEquations, NLsolve, Plots, Statistics, LinearAlgebra, StaticArrays, NLsolve, PGFPlotsX,LaTeXStrings

function flutter_eq_CBC(u, p, t) #Flutter equation of motion
    μ=p.p[1]
    Kp=p.p[2]
    Kd=p.p[3]
    c=p.p[4:end]
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

function flutter_eq_CBC_J(u, p, t) #Jacobian of flutter equation
    μ=p.p[1]
    Kp=p.p[2]
    Kd=p.p[3]
    c=p.p[4:end]
    ind1=1;ind2=3;U=μ
    theta=atan(u[ind1],u[ind2]) #theta is computed from two state variables
    r=c[1]
    h=c[2:end]
    nh=Int(length(h)/2)
    dpd1=u[ind2]/(u[ind1]^2+u[ind2]^2)
    dpd2=-u[ind1]/(u[ind1]^2+u[ind2]^2)
    drd1=0;drd2=0;
    for i=1:nh
        r+=h[i]*cos(i*theta)+h[i+nh]*sin(theta*i)
        drd1+=-h[i]*i*sin(i*theta)*dpd1+h[i+nh]*i*cos(theta*i)*dpd1
        drd2+=-h[i]*i*sin(i*theta)*dpd2+h[i+nh]*i*cos(theta*i)*dpd2
    end
    b=0.15; a=-0.5; rho=1.204; x__alpha=0.2340
    c__0=1; c__1=0.1650; c__2=0.0455; c__3=0.335; c__4=0.3; c__alpha=0.562766779889303; c__h=15.4430
    I__alpha=0.1726; k__alpha=54.116182926744390; k__h=3.5294e+03; m=5.3; m__T=16.9

    MM = [b ^ 2 * pi * rho + m__T -a * b ^ 3 * pi * rho + b * m * x__alpha 0; -a * b ^ 3 * pi * rho + b * m * x__alpha I__alpha + pi * (0.1e1 / 0.8e1 + a ^ 2) * rho * b ^ 4 0; 0 0 1;]
    DD = [c__h + 2 * pi * rho * b * U * (c__0 - c__1 - c__3) (1 + (c__0 - c__1 - c__3) * (1 - 2 * a)) * pi * rho * b ^ 2 * U 2 * pi * rho * U ^ 2 * b * (c__1 * c__2 + c__3 * c__4); -0.2e1 * pi * (a + 0.1e1 / 0.2e1) * rho * (b ^ 2) * (c__0 - c__1 - c__3) * U c__alpha + (0.1e1 / 0.2e1 - a) * (1 - (c__0 - c__1 - c__3) * (1 + 2 * a)) * pi * rho * (b ^ 3) * U -0.2e1 * pi * rho * (U ^ 2) * (b ^ 2) * (a + 0.1e1 / 0.2e1) * (c__1 * c__2 + c__3 * c__4); -1 / b a - 0.1e1 / 0.2e1 (c__2 + c__4) * U / b;]
    KK = [k__h 2 * pi * rho * b * U ^ 2 * (c__0 - c__1 - c__3) 2 * pi * rho * U ^ 3 * c__2 * c__4 * (c__1 + c__3); 0 k__alpha - 0.2e1 * pi * (a + 0.1e1 / 0.2e1) * rho * (c__0 - c__1 - c__3) * (b ^ 2) * (U ^ 2) -0.2e1 * pi * rho * b * (U ^ 3) * (a + 0.1e1 / 0.2e1) * c__2 * c__4 * (c__1 + c__3); 0 -U / b c__2 * c__4 * U ^ 2 / b ^ 2;]
    M2=inv(MM);K1=-M2*KK;D1=-M2*DD;

    J1=[0 1 0 0 0 0]
    J2=[K1[1,1] D1[1,1] K1[1,2] D1[1,2] K1[1,3] D1[1,3]]
    J3=[0 0 0 1 0 0]
    J4=[K1[2,1] D1[2,1] K1[2,2] D1[2,2] K1[2,3] D1[2,3]]
    J5=[0 0 0 0 0 1]
    J6=[K1[3,1] D1[3,1] K1[3,2] D1[3,2] K1[3,3] D1[3,3]]

    J=[J1;J2;J3;J4;J5;J6]
    #Control added on heave
    dcd1=Kp*(drd1*cos(theta)-r*sin(theta)*dpd1-1)+Kd*(drd1*sin(theta)+r*cos(theta)*dpd1)
    dcd2=Kp*(drd2*cos(theta)-r*sin(theta)*dpd2)+Kd*(drd2*sin(theta)+r*cos(theta)*dpd2-1)

    ka2=751.6; ka3=5006;
    dhd2=-M2[1,2]*(2*ka2*u[3]+ka3*3*u[3]^2)
    dtd2=-M2[2,2]*(2*ka2*u[3]+ka3*3*u[3]^2)

    J[2,ind1]+=M2[1,1]*dcd1
    J[4,ind1]+=M2[2,1]*dcd1

    J[2,ind2]+=dhd2+M2[1,1]*dcd2
    J[4,ind2]+=dtd2+M2[2,1]*dcd2
    return J
end

function flutter_eq_CBC_J2(u, p, t) #Jacobian of flutter equation (differentiation with respect to wind speed)
    μ=p.p[1]
    Kp=p.p[2]
    Kd=p.p[3]
    c=p.p[4:end]
    U=μ
    b=0.15; a=-0.5; rho=1.204; x__alpha=0.2340
    c__0=1; c__1=0.1650; c__2=0.0455; c__3=0.335; c__4=0.3; c__alpha=0.562766779889303; c__h=15.4430
    I__alpha=0.1726; k__alpha=54.116182926744390; k__h=3.5294e+03; m=5.3; m__T=16.9

    MM = [b ^ 2 * pi * rho + m__T -a * b ^ 3 * pi * rho + b * m * x__alpha 0; -a * b ^ 3 * pi * rho + b * m * x__alpha I__alpha + pi * (0.1e1 / 0.8e1 + a ^ 2) * rho * b ^ 4 0; 0 0 1;]
    DD = [2 * pi * rho * b  * (c__0 - c__1 - c__3) (1 + (c__0 - c__1 - c__3) * (1 - 2 * a)) * pi * rho * b ^ 2  2 * pi * rho *2 * U * b * (c__1 * c__2 + c__3 * c__4); -0.2e1 * pi * (a + 0.1e1 / 0.2e1) * rho * (b ^ 2) * (c__0 - c__1 - c__3)  (0.1e1 / 0.2e1 - a) * (1 - (c__0 - c__1 - c__3) * (1 + 2 * a)) * pi * rho * (b ^ 3)  -0.2e1 * pi * rho * 2*U * (b ^ 2) * (a + 0.1e1 / 0.2e1) * (c__1 * c__2 + c__3 * c__4); 0 0 (c__2 + c__4) / b;]
    KK = [0 2 * pi * rho * b *2* U  * (c__0 - c__1 - c__3) 2 * pi * rho *3* U ^ 2 * c__2 * c__4 * (c__1 + c__3); 0 0 - 0.2e1 * pi * (a + 0.1e1 / 0.2e1) * rho * (c__0 - c__1 - c__3) * (b ^ 2) * 2*U  -0.2e1 * pi * rho * b * 3*U ^ 2 * (a + 0.1e1 / 0.2e1) * c__2 * c__4 * (c__1 + c__3); 0 -1 / b c__2 * c__4 *2* U / b ^ 2;]
    M2=inv(MM)
    K1=-M2*KK;
    D1=-M2*DD;

    J1=[0 0 0 0 0 0]
    J2=[K1[1,1] D1[1,1] K1[1,2] D1[1,2] K1[1,3] D1[1,3]]
    J3=[0 0 0 0 0 0]
    J4=[K1[2,1] D1[2,1] K1[2,2] D1[2,2] K1[2,3] D1[2,3]]
    J5=[0 0 0 0 0 0]
    J6=[K1[3,1] D1[3,1] K1[3,2] D1[3,2] K1[3,3] D1[3,3]]

    J=[J1;J2;J3;J4;J5;J6]
    #Control added on heave
    J2=J*u
    return J2
end

dimension(::typeof(flutter_eq_CBC)) = 6

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
    # Evaluate the derivative; for-loop for speed equivalent to u*Dᵀ
    for (i, ii) in pairs(1:n:n*N)
        # Evaluate the derivative at the i-th grid point
        for (j, jj) in pairs(1:n:n*N)
            for k = 1:n
                res[ii+k-1] -= p.D[i, j]*u[jj+k-1]
            end
        end
    end
    res[end] = (u[1] - ee)  # phase condition - assumes that the limit cycle passes through the Poincare section u₁=1e-4; using a non-zero value prevents convergence to the trivial solution
    return res
end

function periodic_zero_problem2(rhs, u, p, ee, dv, pu, ds) # Including Pseudo arc-length equation
    # Assume that u has dimensions nN + 1 where n is the dimension of the ODE
n = dimension(rhs)
N = (length(u)-1)÷n
res=Array{Float64}(undef,n*N+2,1)
res1=periodic_zero_problem(rhs, u, p, ee)
p1=p.p[1];
uu=[u;p1]
du=uu-pu

res[1:end-1]=res1
arclength=norm(transpose(dv)*du)
res[end] = arclength-ds  #Pseudo-arclength equation
    return res
end

function periodic_zero_J(jeq, em, u, p, ee)
    # Assume that u has dimensions nN + 1 where n is the dimension of the ODE
    n = dimension(em)
    N = (length(u)-1)÷n
    T = u[end]/(2π) # the differentiation matrix is defined on [0, 2π]
    J = zeros(n*N+1,n*N+1)
    # Evaluate the right-hand side of the ODE
    for i in 1:n:n*N
        J[i:i+n-1,i:i+n-1] = T*jeq(@view(u[i:i+n-1]), p, 0)  # use a view for speed
        J[i:i+n-1,n*N+1] = em(@view(u[i:i+n-1]), p, 0)
    end
    # Evaluate the derivative; for-loop for speed equivalent to u*Dᵀ
    for (i, ii) in pairs(1:n:n*N)
        # Evaluate the derivative at the i-th grid point
        for (j, jj) in pairs(1:n:n*N)
            for k = 1:n
                J[ii+k-1,jj+k-1] -= p.D[i, j]
            end
        end
    end
    J[n*N+1,1:n*N+1] = zeros(1,n*N+1)
    J[n*N+1,1] = 1
      # phase condition
    return J
end

function periodic_zero_J2(jeq, jeq2, em, u, p, ee, dv)
    # Assume that u has dimensions nN + 1 where n is the dimension of the ODE
    # the differentiation matrix is defined on [0, 2π]
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

function LCO_solve(u1,ini_T,eq,jeq,p,ee,z_tol)
u0=[u1;ini_T]
err=0
for i in 1:40
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

function LCO_solve2(u1,ini_T,eq,jeq,jeq2,p,ee,z_tol,dv,pu,ds)
p0=p.p[1]
u0=[u1;ini_T]
uu0=[u0;p0]
err=0
np=p
for i in 1:40
    res=periodic_zero_problem2(eq, u0, np, ee, dv, pu, ds)
    uu1=uu0-periodic_zero_J2(jeq, jeq2, eq, u0, np, ee, dv)\res
    uu0=uu1
    u0=uu0[1:end-1]
    p0=uu0[end]
    np=(p=[p0 transpose(p.p[2:end])],D=p.D)
    er=norm(transpose(res)*res)
    err=hcat(err,er)
    if er<z_tol
        break
    end
end
    return (u=uu0, err=err)
end

function get_sol(u0,dim,N,ind1,ind2)
u=u0[1:end-1]
T=u0[end]
uu=Array{Float64}(undef,N,dim)
theta=Array{Float64}(undef,N)
r=Array{Float64}(undef,N)
for i in 1:dim
    uu[:,i]=u[i:dim:end]
end
for i in 1:N
    theta[i]=atan(uu[i,ind1],uu[i,ind2])
    r[i]=sqrt(uu[i,ind1]^2+uu[i,ind2]^2)
end
    return (u=uu,T=T,t=theta,r=r)
end

function get_sol2(sol,ind)
lu=length(sol.u)
u=Vector{Float64}(undef, lu)
for i in 1:lu
    uv=sol.u[i]
    u[i]=uv[ind]
end
return u
end

function get_stable_LCO(p,u0,tl,tol,eq,stol,rp) # Get a stable LCO from numerical integration
u=u0
dim=length(u0)
prob = ODEProblem(eq,u,(0,tl*rp),p)
sol = solve(prob,Tsit5(),reltol=stol,abstol=stol)
vP=1
P=0
T=0
while vP>tol
u=sol.u[end]
prob = ODEProblem(eq,u,(0,tl),p)
sol = solve(prob,Tsit5(),reltol=stol,abstol=stol)
z=zero_measure(sol.u,1,sol.t)
vP=var(z.hp)
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
ind1=1;ind2=3;u=uu[:,ind1];v=uu[:,ind2]
for i in 1:length(u)
    t[i]=atan(u[i],v[i])
    r[i]=sqrt(u[i]^2+v[i]^2)
end
return (u=uu,t=t,r=r,P=P,T=T)
end

function picard_CBC(u1,ini_T,eq,jeq,p,ee,z_tol,c_tol)
p0=p.p[1:3]
c0=p.p[4:end]
D0=p.D
nh=Int((length(c0)-1)/2)
dm=dimension(eq)
N=Int(length(u1)/dm)
s=LCO_solve(u1,ini_T,eq,jeq,p,ee,z_tol)
g=get_sol(s.u,6,N,1,3)
r=LS_harmonics(g.r,g.t,1,nh)
c=r.coeff
del_c=c-c0
err=transpose(del_c)*del_c
err=sqrt(err)
e=1
p= (p=[transpose(p0) transpose(c)],D=D0)
while err>c_tol
    c0=p.p[4:end]
    nh=Int((length(c0)-1)/2)
    s=LCO_solve(u1,ini_T,eq,jeq,p,ee,z_tol)
    g=get_sol(s.u,6,N,1,3)
    r=LS_harmonics(g.r,g.t,1,nh)
    c=r.coeff
    del_c=c-c0
    err=sqrt(transpose(del_c)*del_c)
    p= (p=[transpose(p0) transpose(c)],D=D0)
    e=vcat(e,err)
end
return (s=s,p=p,e=e)
end

function Newton_CBC(u1,ini_T,eq,jeq,p,ee,z_tol,c_tol,ind1,ind2)
p0=p.p[1:3]
c0=p.p[4:end]
D0=p.D
nh=Int((length(c0)-1)/2)
dm=dimension(eq)
N=Int(length(u1)/dm)
s=LCO_solve(u1,ini_T,eq,jeq,p,ee,z_tol)
g=get_sol(s.u,6,N,ind1,ind2)
r=LS_harmonics(g.r,g.t,1,nh)
c=r.coeff
del_c0=c-c0
err=transpose(del_c0)*del_c0
er=err
for i in 1:10
    if err<c_tol
        break
    end
    J=Jac_CBC(p,u1,ini_T,eq,jeq,ee,ind1,ind2,z_tol)
    del_c=inv(J)*del_c0
    c1=c0-del_c
    p= (p=[transpose(p0) transpose(c1)],D=D0)
    s=LCO_solve(u1,ini_T,eq,jeq,p,ee,z_tol)
    g=get_sol(s.u,6,N,ind1,ind2)
    r=LS_harmonics(g.r,g.t,1,nh)
    del_c1=r.coeff-c1
    del_c0=del_c1
    c0=c1
    err=transpose(del_c1)*del_c1
    er=vcat(er,err)
end
return (s=s,p=p,e=er)
end

function Newton_CBC(u1,ini_T,eq,jeq,p,ee,z_tol,c_tol,ind1,ind2,dv,pu,ds)
p0=p.p[1:3]
c0=p.p[4:end]
nh=Int((length(c0)-1)/2)
dm=dimension(eq)
N=Int(length(u1)/dm)
rr=res_CBC(p,u1,ini_T,eq,jeq,ee,ind1,ind2,z_tol,dv,pu,ds)
res=rr.res
u1=rr.u1;ini_T=rr.T1;
err=transpose(res)*res
er=err
for i in 1:10
    if err<c_tol
        break
    end
    J=Jac_CBC(p,u1,ini_T,eq,jeq,ee,ind1,ind2,z_tol,dv)
    del_u=inv(J)*res
    c1=c0-del_u[1:end-1]
    p0[1]=p0[1]-del_u[end]
    p= (p=[transpose(p0) transpose(c1)],D=p.D)
    rr=res_CBC(p,u1,ini_T,eq,jeq,ee,ind1,ind2,z_tol,dv,pu,ds)
    res=rr.res
    u1=rr.u1;ini_T=rr.T1;
    c0=c1
    err=transpose(res)*res
    er=vcat(er,err)
end
return (u1=u1,T1=ini_T,p=p,e=er)
end

function Jac_CBC(p,u0,ini_T,eq,jeq,ee,ind1,ind2,z_tol)
pp=p.p[1:3]
c0=p.p[4:end]
dm=dimension(eq)
nh=Int((length(c0)-1)/2)
N=Int(length(u0)/dm)
s=LCO_solve(u0,ini_T,eq,jeq,p,ee,z_tol)
g=get_sol(s.u,dm,N,ind1,ind2)
c=LS_harmonics(g.r,g.t,1,nh).coeff
del_c0=c-c0
J=Array{Float64}(undef,2*nh+1,2*nh+1)
eye=SMatrix{2*nh+1,2*nh+1}(1I)
delta=1e-9
for i in 1:2*nh+1
    c=c0+eye[i,:]*delta
    np=(p=[transpose(pp) transpose(c)],D=p.D)
    s=LCO_solve(u0,ini_T,eq,jeq,np,ee,z_tol)
    g=get_sol(s.u,dm,N,ind1,ind2)
    dc=LS_harmonics(g.r,g.t,1,nh).coeff
    del_c1=dc-c
    dp=(del_c1-del_c0)/delta
    J[:,i]=dp
end
return J
end

function Jac_CBC(p,u0,ini_T,eq,jeq,ee,ind1,ind2,z_tol,dv) #Jacobian for CBC with Pseudo arclength
pp=p.p[1:3]
c0=p.p[4:end]
dm=dimension(eq)
nh=Int((length(c0)-1)/2)
N=Int(length(u0)/dm)
s=LCO_solve(u0,ini_T,eq,jeq,p,ee,z_tol)
g=get_sol(s.u,dm,N,ind1,ind2)
c=LS_harmonics(g.r,g.t,1,nh).coeff
del_c0=c-c0
J=Array{Float64}(undef,2*nh+2,2*nh+2)
eye=SMatrix{2*nh+1,2*nh+1}(1I)
delta=1e-8
for i in 1:2*nh+1
    c=c0+eye[i,:]*delta
    np=(p=[transpose(pp) transpose(c)],D=p.D)
    s=LCO_solve(u0,ini_T,eq,jeq,np,ee,z_tol)
    g=get_sol(s.u,dm,N,ind1,ind2)
    dc=LS_harmonics(g.r,g.t,1,nh).coeff
    del_c1=dc-c
    dp=(del_c1-del_c0)/delta
    J[1:end-1,i]=dp
end
npp=pp
npp[1]=npp[1]+delta
np=(p=[transpose(npp) transpose(c0)],D=p.D)
s=LCO_solve(u0,ini_T,eq,jeq,np,ee,z_tol)
g=get_sol(s.u,dm,N,ind1,ind2)
dc=LS_harmonics(g.r,g.t,1,nh).coeff
del_c1=dc-c0
dp=(del_c1-del_c0)/delta
J[1:end-1,end]=dp
J[end,:]=dv
return J
end

function res_CBC(p,u0,ini_T,eq,jeq,ee,ind1,ind2,z_tol,dv,pu,ds) #Jacobian for CBC with Pseudo arclength
pp=p.p[1:3]
c0=p.p[4:end]
dm=dimension(eq)
nh=Int((length(c0)-1)/2)
N=Int(length(u0)/dm)
res=Array{Float64}(undef,2*nh+2,1)
s=LCO_solve(u0,ini_T,eq,jeq,p,ee,z_tol)
g=get_sol(s.u,dm,N,ind1,ind2)
c=LS_harmonics(g.r,g.t,1,nh).coeff
del_c0=c-c0
uu=[c0;pp[1]]
du=uu-pu
res[1:end-1]=del_c0
#arclength=norm(dv*transpose(du))
arclength=dv*transpose(du)
arclength=arclength[1]
res[end] = arclength-ds
return (res=vec(res),u1=s.u[1:end-1],T1=s.u[end])
end

function zero_measure(u,ind,t)
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

function MonoD(u,p,t)
n=6
v=u[1:Int(n*n)]
uu=u[Int(n*n+1):end]
M=reshape(v, n, n)
J=flutter_eq_CBC_J(uu,p,t)
dM=J*M
dv=vec(dM)
duu=flutter_eq_CBC(uu,p,t)
duu=vec(duu)
du=[dv;duu]
 return du
end

function Monodromy_compute(u,p,dim,N)
n=dim
eye=1*Matrix(I, n, n)
v0=vec(eye)
g=get_sol(u,dim,N,1,3)
T=g.T
uu=g.u
M=eye
tl2=T/N
for i in 1:N
    uu0=uu[i,:]
    u0=[v0;uu0]
    prob = ODEProblem(MonoD,u0,(0,tl2),p)
    sol = solve(prob,Tsit5(),reltol=1e-14,abstol=1e-14)
    w=sol.u[end]
    m1=w[1:Int(n*n)]
    M1=reshape(m1, n, n)
    M=M1*M
end
Eig1=eigen(M)
μ=Eig1.values
 return μ
end

function LS_harmonics(r,t,ω,N)
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

function ini_val(p,tl,tol,N,eq,u0,stol,rp) #Get stable LCO from the ODE numerical solutions- initial collocation points
dim=length(u0)
s0=get_stable_LCO(p,u0,tl,tol,eq,stol,rp)
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
eq=flutter_eq_CBC;
#uu=pu+ds*dv;
for i in 1:N-1
    prob = ODEProblem(eq,u0,(0,tl2),p)
    sol = solve(prob,Tsit5(),reltol=1e-11,abstol=1e-11)
    u0=sol.u[end]
    ini_u[Int(dim*i+1):Int(dim*i+dim)]=u0
end
    return (u=ini_u,T=ini_T)
end

function get_CBC_u(p) #Get stable LCO from the ODE numerical solutions- initial collocation points
c0=p.p[4:end]
nu=p.p[1]
u=[c0;nu]
    return u
end

function get_CBC_p(nu,Kp,Kd,N) #Get stable LCO from the ODE numerical solutions- initial collocation points
c=nu[1:end-1]
p=(p=[nu[end] Kp Kd transpose(c)],D=fourier_diff(N))
    return p
end

function get_dv_u(u,T1,pu,ds) #Get stable LCO from the ODE numerical solutions- initial collocation points
uu=[u;T1]
dv=uu-pu;dv=dv/norm(dv)
nu=uu+dv*ds
u1=nu[1:end-1]
T1=nu[end]
    return (u=u1,T=T1)
end

function amp_LCO(U,ind) #Get stable LCO from the ODE numerical solutions- initial collocation points
l1=length(U)
amp=Vector{Float64}(undef, l1)
for i in 1:l1
    g=get_sol(U[i],6,N,1,3)
    uu=g.u[:,ind]
    amp[i]=maximum(uu)-minimum(uu)
end
    return amp
end

function continuation_flutter(tol,N,sp,sU,ds) #Get stable LCO from the ODE numerical solutions- initial collocation points
Kp=0;Kd=0;μ=sU+0.02;p= (p=[μ 0 0 0 0 0],D=fourier_diff(N));ee=1e-3;
eq=flutter_eq_CBC;tl=2.0;tol=1e-5;jeq=flutter_eq_CBC_J;jeq2=flutter_eq_CBC_J2;
stol=1e-6;rp=3;u0=[0.1;0.1;0.1;0;0;0];tl=3.0
s1=ini_val(p,tl,tol,N,eq,u0,stol,rp)
s=LCO_solve(s1.u,s1.T,eq,jeq,p,s1.u[1],tol)

μ=sU;p= (p=[μ 0 0 0 0 0],D=fourier_diff(N));
s2=LCO_solve(s1.u,s1.T,eq,jeq,p,s1.u[1],tol)

du=s2.u-s.u
du=[du;-0.02]
dv=du/norm(du)
dim=6;ll=dim*N+1;
V = [zeros(ll) for _ in 1:sp]
P = zeros(sp,1)
pu=s2.u;pu=[pu;sU]
V[1]=vec(s2.u);P[1]=sU

uu=pu+ds*dv
μ=uu[end];p= (p=[μ 0 0 0 0 0],D=fourier_diff(N));u=uu[1:end-1];
u1=u[1:end-1];ini_T=u[end];z_tol=tol*0.1;
for i in 2:sp
    s2=LCO_solve2(u1,ini_T,eq,jeq,jeq2,p,ee,z_tol,dv,pu,ds)
    s2.err
    du=s2.u-pu;dv=du/norm(du)
    V[i]=s2.u[1:end-1];P[i]=s2.u[end]
    pu=s2.u;
    uu=pu+ds*dv;
    p= (p=[uu[end] 0 0 0 0 0],D=fourier_diff(N));u=uu[1:end-1];
    u1=u[1:end-1];ini_T=u[end];z_tol=tol*0.1;
end
    return (V=V,P=P)
end


function continuation_CBC(tol,ctol,N,sp,sU,ds,nh) #Get stable LCO from the ODE numerical solutions- initial collocation points
# Compute initial points
l1=nh*2+1;l2=6*N+1
H = [zeros(l1) for _ in 1:sp]
U = [zeros(l2) for _ in 1:sp]
P = zeros(sp,1)

Kp=-100;Kd=-100
eq=flutter_eq_CBC;jeq=flutter_eq_CBC_J;
jeq2=flutter_eq_CBC_J2;
s0=continuation_flutter(tol,N,10,sU,0.1)

lp=9;s0.P[lp]
uu=s0.V[lp];p0= (p=[s0.P[lp] 0 0 0 0 0],D=fourier_diff(N))
g=get_sol(uu,6,N,1,3)
d=LS_harmonics(g.r,g.t,1,nh)
c=transpose(d.coeff)
p1= (p=[s0.P[lp] Kp Kd c],D=fourier_diff(N))
s=LCO_solve(uu[1:end-1],uu[end],eq,jeq,p1,1e-3,tol)
cbc_u1=get_CBC_u(p1)

lp=10;s0.P[lp]
uu=s0.V[lp];p0= (p=[s0.P[lp] 0 0 0 0 0],D=fourier_diff(N))
g=get_sol(uu,6,N,1,3)
d=LS_harmonics(g.r,g.t,1,nh)
c=transpose(d.coeff)
p2= (p=[s0.P[lp] Kp Kd c],D=fourier_diff(N))
s2=LCO_solve(uu[1:end-1],uu[end],eq,jeq,p2,1e-3,tol)
cbc_u2=get_CBC_u(p2)

dv=cbc_u2-cbc_u1;dv=dv/norm(dv);pu=cbc_u2;
aa=-dv[end]
dv=vcat(dv[1:end-1],aa)
if s0.P[2]>s0.P[1]
    dv=-dv;pu=cbc_u1;
end
ds2=0.001
nu=pu+ds2*dv;c=nu[1:end-1]
p=get_CBC_p(nu,Kp,Kd,N)
dvu=s2.u-s.u;dvu=dvu/norm(dvu);
cpu=s2.u;ncu=cpu+dvu*ds
u1=ncu[1:end-1];T1=ncu[end]
ee=1e-3;z_tol=1e-8;
cs=Newton_CBC(u1,T1,eq,jeq,p,ee,z_tol,ctol,1,3,dv,pu,ds)
P[1]=cs.p.p[1]
H[1]=cs.p.p[4:end]
U[1]=[cs.u1;cs.T1]
dv=get_CBC_u(cs.p)-pu;dv=dv/norm(dv);
pu=get_CBC_u(cs.p);
nu=pu+ds*dv;p=get_CBC_p(nu,Kp,Kd,N)
uu=get_dv_u(cs.u1,cs.T1,cpu,ds)
u1=uu.u;T1=uu.T;

for i in 2:sp
    cs=Newton_CBC(u1,T1,eq,jeq,p,ee,z_tol,ctol,1,3,dv,pu,ds)
    P[i]=cs.p.p[1]
    H[i]=cs.p.p[4:end]
    U[i]=[cs.u1;cs.T1]
    dv=get_CBC_u(cs.p)-pu;dv=dv/norm(dv);
    pu=get_CBC_u(cs.p);
    nu=pu+ds*dv;p=get_CBC_p(nu,Kp,Kd,N)
    uu=get_dv_u(cs.u1,cs.T1,cpu,ds)
    u1=uu.u;T1=uu.T;
end
    return (H=H,U=U,P=P)
end


# Continuation of Equation of motion
sU=17.0;sp=130;N=100;tol=1e-8;ds=0.1;
eq=flutter_eq_CBC;u0=[0.1;0.001;0.001;0;0;0];tl=2.0;tol=1e-5;jeq=flutter_eq_CBC_J;
jeq2=flutter_eq_CBC_J2;
@time s0=continuation_flutter(tol,N,sp,sU,ds)
amp1=amp_LCO(s0.V,1)
p1=vec(s0.P)
# Continuation of CBC problem
sp=30;tol=1e-7;ctol=1e-6;N=100;sU=15.1;ds=0.01;nh=10;
@time s_cbc=continuation_CBC(tol,ctol,N,sp,sU,ds,nh)
U=s_cbc.U
P=s_cbc.P
amp=amp_LCO(U,1)
P=vec(P)

#Bifurcation diagram plot
a=@pgf Axis( {xlabel="Wind speed (m/s)",
            ylabel = "Heave amplitude (rad)",
            legend_pos  = "north west",
            height="9cm",
            width="13cm",ymin=0,ymax=0.08},

    Plot(
        { color="red",
            no_marks
        },
        Coordinates(p1,amp1)
    ),
    LegendEntry("Numerical continuation"),
    Plot(
        { color="blue",
            mark="*",
        },
        Coordinates(P,amp)
    ),
    LegendEntry("CBC")
)

pgfsave("cbc_cont.pdf",a)

lp=10
uu=s_cbc.U[lp];p0= (p=[s_cbc.P[lp] 0 0 0 0 0],D=fourier_diff(N))
jeq=flutter_eq_CBC_J;jeq2=flutter_eq_CBC_J2;
s=LCO_solve(uu[1:end-1],uu[end],eq,jeq,p0,1e-3,1e-11)

g=get_sol(s.u,6,N,1,3)
g2=get_sol(uu,6,N,1,3)

nu1=Monodromy_compute(s.u,p0,6,N)
r1=real(nu1)
i1=imag(nu1)
c=transpose(s_cbc.H[lp])
Kp=-100;Kd=-100;p= (p=[s_cbc.P[lp] Kp Kd c],D=fourier_diff(N))
nu2=Monodromy_compute(s_cbc.U[lp],p,6,N)
r2=real(nu2)
i2=imag(nu2)

a=@pgf Axis( {xlabel=L"$h$ (m)",
            ylabel = L"$\alpha$ (rad)",
            legend_pos  = "north west",
            height="9cm",
            width="9cm",
            xmin=-2e-2,
            xmax=2e-2,
            ymin=-7e-2,
            ymax=6e-2},

    Plot(
        { color="red",
            no_marks
        },
        Coordinates(vcat(g.u[:,1],g.u[1,1]),vcat(g.u[:,3],g.u[1,3]))
    ),
    LegendEntry("Numerical continuation"),
    Plot(
        { color="blue",
            no_marks,
        },
        Coordinates(vcat(g2.u[:,1],g.u[1,1]),vcat(g2.u[:,3],g.u[1,3]))
    ),
    LegendEntry("CBC")
)
pgfsave("ppcbc_16.pdf",a)
#Monodromy matrix

xx=[cos(θ) for θ in range(0,2π,length=100)]
yy=[sin(θ) for θ in range(0,2π,length=100)]

a=@pgf Axis(
    {        height="10cm",
            width="10cm",
            xlabel="Real",
            ylabel="Imaginary",
            legend_pos  = "north west",
        "scatter/classes" = {
            a = {mark = "o", "blue"},
            b = {mark = "triangle*", "red"},
        }
    },
    Plot(
        {
            scatter,
            "only marks",
            "scatter src" = "explicit symbolic",
                        mark_options = {scale=1.7}
        },
        Table(
            {
                meta = "label"
            },
            x = r1,
            y = i1,
            label = ["a", "a", "a", "a", "a", "a"],
        )
    ),
        LegendEntry("Numerical continuation"),
        Plot(
            {
                scatter,
                "only marks",
                "scatter src" = "explicit symbolic",
                            mark_options = {scale=1.7}
            },
            Table(
                {
                    meta = "label"
                },
                x = r2,
                y = i2,
                label = ["b", "b", "b", "b", "b", "b"],
            )
        ),
            LegendEntry("CBC"),
        Plot(
            { color="black",
                no_marks
            },
            Coordinates(xx,yy)
        ),
)
pgfsave("monodroby_16.pdf",a)
lp=20
uu=s_cbc.U[lp];p0= (p=[s_cbc.P[lp] 0 0 0 0 0],D=fourier_diff(N))
jeq=flutter_eq_CBC_J;jeq2=flutter_eq_CBC_J2;
s=LCO_solve(uu[1:end-1],uu[end],eq,jeq,p0,1e-3,1e-11)

g=get_sol(s.u,6,N,1,3)
g2=get_sol(uu,6,N,1,3)

nu1=Monodromy_compute(s.u,p0,6,N)
r1=real(nu1)
i1=imag(nu1)
c=transpose(s_cbc.H[lp])
Kp=-100;Kd=-100;p= (p=[s_cbc.P[lp] Kp Kd c],D=fourier_diff(N))
nu2=Monodromy_compute(s_cbc.U[lp],p,6,N)
r2=real(nu2)
i2=imag(nu2)

a=@pgf Axis( {xlabel=L"$h$ (m)",
            ylabel = L"$\alpha$ (rad)",
            legend_pos  = "north west",
            height="9cm",
            width="9cm",
            xmin=-1.5e-2,
            xmax=1.5e-2,
            ymin=-5e-2,
            ymax=4.5e-2},

    Plot(
        { color="red",
            no_marks
        },
        Coordinates(vcat(g.u[:,1],g.u[1,1]),vcat(g.u[:,3],g.u[1,3]))
    ),
    LegendEntry("Numerical continuation"),
    Plot(
        { color="blue",
            no_marks,
        },
        Coordinates(vcat(g2.u[:,1],g.u[1,1]),vcat(g2.u[:,3],g.u[1,3]))
    ),
    LegendEntry("CBC")
)
pgfsave("ppcbc_17.pdf",a)

#Monodromy matrix

a=@pgf Axis(
    {        height="10cm",
            width="10cm",
            xlabel="Real",
            ylabel="Imaginary",
            legend_pos  = "north west",
        "scatter/classes" = {
            a = {mark = "o", "blue"},
            b = {mark = "triangle*", "red"},
        }
    },
    Plot(
        {
            scatter,
            "only marks",
            "scatter src" = "explicit symbolic",
                        mark_options = {scale=1.7}
        },
        Table(
            {
                meta = "label"
            },
            x = r1,
            y = i1,
            label = ["a", "a", "a", "a", "a", "a"],
        )
    ),
        LegendEntry("Numerical continuation"),
        Plot(
            {
                scatter,
                "only marks",
                "scatter src" = "explicit symbolic",
                            mark_options = {scale=1.7}
            },
            Table(
                {
                    meta = "label"
                },
                x = r2,
                y = i2,
                label = ["b", "b", "b", "b", "b", "b"],
            )
        ),
            LegendEntry("CBC"),
        Plot(
            { color="black",
                no_marks
            },
            Coordinates(xx,yy)
        ),
)
pgfsave("monodromy_17.pdf",a)
