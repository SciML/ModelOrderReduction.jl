using Symbolics

# polynomial transformations
# given two polynomial bases ξ₁(x) and ξ₂(x), we identify both polynomials simply by 
# p(x) = a'ξ₁(x) = b'ξ₂(x)
# often we wish to compute b from a. 
# to do so we only need to find T such that Tξ₁ = ξ₂.
# A convenient way is to expand both ξ₁ and ξ₂ in the monomial basis: Aξ₁ = Bξ₂ = ξ₃ 
# Then T = B⁻¹A. 
# Typically one chooses the monomial basis as A and B are remarkably easy to find in that case.
# That, however, comes at the cost of poor numerical conditioning of the A and B and hence potential
# issues arise in the computation of T = B⁻¹A. 

# For PCE a natural workflow is obtained as
# give f(x,p,t) = [f1(x,p,t), ..., fN(x,p,t)]
# choose polynomial basis ξ(p) = [ξ1(p1), ..., ξK(p1)] ⨂ ⋯ ⨂ [ξ1(pL), ..., ξK(pL)]   => this contains redundant terms >:(
# choose solution ansatz x(p,t) = [x1(p,t), ..., xN(p,t)] = C(t)ξ(p)
# by asumption dC(t)/dt ξ(p) = f(C(t)ξ(p), p, t)
# => ∫ dC(t)/dt ξ(p) ξ'(p) dρ(p) = dC(t)/dt ∫ ξ(p) ξ'(p) dρ(p) = ∫ f(C(t)ξ(p), p, t) ξ'(p) dρ(p)
# If ξ(p) is orthogonal with respect to ρ(p) then ∫ ξ(p) ξ'(p) dρ(p) = I
# so dC(t)/dt = ∫ f(C(t)ξ(p), p, t) ξ'(p) dρ(p)
# We therefore need to evaluate the right-hand-side. 
# In general this is very hard. Here we are going to make this easy by assuming that 
# fi(x,p,t) = (f̂i)' (qi(x) ⨂ ri(p))
# with qi(x) being itself a polynomial basis 
# then we evaluate  Wi(C(t)) vi(p) = qi(C(t)ξ(p)) ⨂ ri(p) 
# and transform vi(p) = [Gi, Git] [ξ(p); ξt(p)] to arrive at
# fi(x,p,t) = (f̂i)'Wi(C(t))[Gi, Git] [ξ(p); ξt(p)]
# then ∫ fi(C(t)ξ(p), p, t) ξ'(p) dρ(p) = (f̂i)'Wi(C(t))[Gi, Git] ∫ [ξ(p); ξt(p)] ξ'(p) dρ(p)
# where ∫ [ξ(p); ξt(p)] ξ'(p) dρ(p) = [I; 0]
# so that eᵢ'dC/dt = (f̂i)'Wi(C(t)) Gi (ith row)

