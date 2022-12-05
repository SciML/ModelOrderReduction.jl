"""
$(TYPEDEF)

An object used as the field `metadata` in `ModelingToolkit.AbstractSystem` to store the
information about the Polynomial Chaos Expansion of a stochastic system.

# Fields
$(TYPEDFIELDS)
"""
struct PCEMetadata{S <: ModelingToolkit.AbstractSystem}
    "A PCE object."
    pce::PCE
    "The original stochastic system on which PCE is applied."
    sys::S
end
