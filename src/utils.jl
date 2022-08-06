"""
$(TYPEDSIGNATURES)

Return the differential equations and other non-differential equations.

For differential equations, this function assume the derivative is the only single term at
the left hand side, which is typically the result of `ModelingToolkit.structural_simplify`.

Equations from subsystems are not included.
"""
function get_deqs(sys::ODESystem)::Tuple{Vector{Equation}, Vector{Equation}}
    eqs = ModelingToolkit.get_eqs(sys)
    deqs = Equation[]
    others = Equation[]
    for eq in eqs
        if eq.lhs isa Term && operation(eq.lhs) isa Differential
            push!(deqs, eq)
        else
            push!(others, eq)
        end
    end
    deqs, others
end
