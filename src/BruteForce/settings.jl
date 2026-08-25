Base.@kwdef mutable struct BruteForceSettings <: AbstractSettings
    max_time::Float64 = 3600.0
    scaled::Bool = false
    print_level::Int = 0
end
