using Test

using CutNorm
using LinearAlgebra
using NLPModels
using SolverCore
using DelimitedFiles

@testset "Sanity Check" begin
    A = [1.0 1 1 1 1;
        1 1 1 1 1;
        1 1 1 1 1;
        -1 -1 -1 -1 -1;
        -1 -1 -1 -1 -1]
    # cut norm = 15, S = [1,1,1,0,0], T = [1,1,1,1,1]
    solution = cutnorm(A; print_level=0)

    @test solution.value ≈ 15.0
    @test solution.S ≈ [1.0, 1.0, 1.0, 0.0, 0.0]
    @test solution.T ≈ [1.0, 1.0, 1.0, 1.0, 1.0]

    solution = cutnorm(A; method=MultistartAugmented{TronSolver}(), scaled=true, save_all_solutions=true, print_level=1, max_restarts=100)
    @test solution.value ≈ 15.0 / 25
    @test solution.S ≈ [1.0, 1.0, 1.0, 0.0, 0.0]
    @test solution.T ≈ [1.0, 1.0, 1.0, 1.0, 1.0]
    @test length(solution.all_solutions) == solution.restarts

    solution = cutnorm(A; method=MultistartSigned{TronSolver}(), scaled=true, save_all_solutions=true, print_level=1, max_restarts=100)
    @test solution.value ≈ 15.0 / 25
    @test solution.S ≈ [1.0, 1.0, 1.0, 0.0, 0.0]
    @test solution.T ≈ [1.0, 1.0, 1.0, 1.0, 1.0]
    @test length(solution.all_solutions) == 2 * solution.restarts

    solution = cutnorm(A; method=MultistartAugmented{AlternatingLinearSearch}(), scaled=true, save_all_solutions=true, print_level=2, max_restarts=100)
    @test solution.value ≈ 15.0 / 25
    @test solution.S ≈ [1.0, 1.0, 1.0, 0.0, 0.0]
    @test solution.T ≈ [1.0, 1.0, 1.0, 1.0, 1.0]
    @test length(solution.all_solutions) == solution.restarts
    solution = cutnorm(A; method=MultistartSigned{AlternatingLinearSearch}(), scaled=true, save_all_solutions=true, print_level=3, max_restarts=50)
    @test solution.value ≈ 15.0 / 25
    @test solution.S ≈ [1.0, 1.0, 1.0, 0.0, 0.0]
    @test solution.T ≈ [1.0, 1.0, 1.0, 1.0, 1.0]
    @test length(solution.all_solutions) == 2 * solution.restarts
end

include("augment_matrix.jl")

include("alternating_linear_search.jl")

include("greedy_solver.jl")

include("bilinear_model.jl")

include("signed_bilinear_model.jl")

include("brute_force.jl")

include("inlp.jl")

include("ilp.jl")

include("qubo.jl")