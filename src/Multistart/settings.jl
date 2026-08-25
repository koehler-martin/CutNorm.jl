Base.@kwdef mutable struct MultistartSettings <: AbstractSettings
    max_restarts::Int = 1000
    max_time::Float64 = 3600.0
    scaled::Bool = false
    save_all_solutions::Bool = false
    print_level::Int = 0
end