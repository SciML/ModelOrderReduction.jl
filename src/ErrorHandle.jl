function errorhandle(data::AbstractMatrix, nmodes::Int, max_energy, min_nmodes, max_nmodes)
    @assert size(data, 1) > 1 "State vector is expected to be vector valued."
    s = minimum(size(data))
    @assert 0 < nmodes <= s "Number of modes should be in {1,2,...,$s}."
    @assert min_nmodes <= max_nmodes "Minimum number of modes must lie below maximum number of modes"
    @assert 0.0 <= max_energy <= 1.0 "Maxmimum relative energy must be in [0,1]"
end

function errorhandle(data::AbstractVector{T}, nmodes::Int, max_energy, min_nmodes, max_nmodes) where {T <: AbstractVector}
    @assert size(data[1], 1) > 1 "State vector is expected to be vector valued."
    s = min(size(data, 1), size(data[1], 1))
    @assert 0 < nmodes <= s "Number of modes should be in {1,2,...,$s}."
    @assert min_nmodes <= max_nmodes "Minimum number of modes must lie below maximum number of modes"
    @assert 0.0 <= max_energy <= 1.0 "Maxmimum relative energy must be in [0,1]"
end