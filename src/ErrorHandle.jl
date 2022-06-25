function errorhandle(data::Matrix{FT},modes::IT) where {FT,IT}
    @assert size(data,1)>1 "State vector is expected to be vector valued."
    s = size(data,2)
    @assert (modes>0)&(modes<s) "Number of modes should be [1,$(s)]."
end
