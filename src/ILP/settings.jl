Base.@kwdef mutable struct ILPSettings <: AbstractSettings
    max_time::Float64 = 3600.0
    scaled::Bool = false
    print_level::Int = 0
end
