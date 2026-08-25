abstract type AbstractSolver{T<:AbstractFloat} end

abstract type AbstractSolution{T<:AbstractFloat} end

abstract type AbstractSettings end

"""
    populate!(s::AbstractSettings; kwargs...) -> AbstractSettings

Change the fields of the `AbstractSettings` object `s` in-place with the values
given through keyword arguments and return `s`.

Only existing fields are accepted; an `ArgumentError` is thrown if an
unknown keyword is supplied. Each value is automatically converted to
the field's declared type.
"""
function populate!(s::S; kwargs...) where {S<:AbstractSettings}
    for (name, val) in pairs(kwargs)
        if !hasfield(S, name)
            throw(ArgumentError("unknown setting: $name. Allowed keywords: $(fieldnames(S))"))
        end
        setproperty!(s, name, val)
    end
    return s
end