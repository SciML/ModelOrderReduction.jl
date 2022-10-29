function stochastic_galerkin(sys::ODESystem, pce::PCE;
                             name::Symbol = Symbol(nameof(sys), :_pce), kwargs...)
    iv = ModelingToolkit.get_iv(sys)
    eqs = full_equations(sys; kwargs...)
end
