@testset "BruteForce Solver" begin
    # 5x5 sanity check matrix from the main test suite
    A = [1.0 1 1 1 1;
        1 1 1 1 1;
        1 1 1 1 1;
        -1 -1 -1 -1 -1;
        -1 -1 -1 -1 -1]

    sol = cutnorm(A; method=BruteForce())
    @test sol.value ≈ 15.0
    @test sol.S ≈ [1.0, 1.0, 1.0, 0.0, 0.0]
    @test sol.T ≈ [1.0, 1.0, 1.0, 1.0, 1.0]
    @test sol.termination_status == :optimal
    @test sol.iterations == 2^5 * 2^5

    # Scaled mode
    sol_scaled = cutnorm(A; method=BruteForce(), scaled=true)
    @test sol_scaled.value ≈ 15.0 / 25

    # All-ones 3x2
    A = [1.0 1.0;
        1.0 1.0;
        1.0 1.0]

    sol = cutnorm(A; method=BruteForce())
    @test sol.value ≈ 6.0
    @test sol.S ≈ [1.0, 1.0, 1.0]
    @test sol.T ≈ [1.0, 1.0]

    # Dominant entry 3x2
    A = [10.0 -1.0;
        -1.0 -1.0;
        -1.0 -1.0]

    sol = cutnorm(A; method=BruteForce())
    @test sol.value ≈ 10.0
    @test sol.S ≈ [1.0, 0.0, 0.0]
    @test sol.T ≈ [1.0, 0.0]
    @test abs(dot(sol.S, A, sol.T)) ≈ sol.value

    # 1x1 matrix
    A = reshape([7.0], 1, 1)

    sol = cutnorm(A; method=BruteForce())
    @test sol.value ≈ 7.0
    @test sol.iterations == 4

    # Negative matrix: cut norm uses absolute value
    A = [-3.0 -4.0; -5.0 -6.0]

    sol = cutnorm(A; method=BruteForce())
    @test sol.value ≈ 18.0
    @test abs(dot(sol.S, A, sol.T)) ≈ sol.value

    # Off-diagonal dominant
    A = [1.0 -2.0; -2.0 1.0]

    sol = cutnorm(A; method=BruteForce())
    @test sol.value ≈ 2.0

    # Zero matrix
    A = zeros(3, 3)

    sol = cutnorm(A; method=BruteForce())
    @test sol.value ≈ 0.0
    @test sol.termination_status == :optimal
end

@testset "BruteForce Direct API" begin
    A = [1.0 2.0; 3.0 4.0]

    solver = BruteForceSolver(A)
    sol = BruteForceSolution(A)
    solve!(solver, sol)

    @test sol.value ≈ 10.0
    @test sol.dims == (2, 2)
    @test sol.termination_status == :optimal
    @test sol.runtime > 0.0

    # Reuse solver with reset
    sol2 = BruteForceSolution(A)
    solve!(solver, sol2)
    @test sol2.value ≈ sol.value
end

@testset "BruteForce Time Limit" begin
    A = randn(12, 12)

    sol = cutnorm(A; method=BruteForce(), max_time=0.0)
    @test sol.termination_status == :max_time
    @test sol.iterations < 2^12 * 2^12
end

@testset "BruteForce vs Multistart" begin
    A = [1.0 1 1 1 1;
        1 1 1 1 1;
        1 1 1 1 1;
        -1 -1 -1 -1 -1;
        -1 -1 -1 -1 -1]

    bf = cutnorm(A; method=BruteForce())
    ms = cutnorm(A; method=MultistartSigned{AlternatingLinearSearch}(), max_restarts=500)

    @test bf.value ≈ ms.value

    A = [10.0 -1.0;
        -1.0 -1.0;
        -1.0 -1.0]

    bf = cutnorm(A; method=BruteForce())
    ms = cutnorm(A; method=MultistartSigned{GreedySolver}(), max_restarts=500)

    @test bf.value ≈ ms.value
end

@testset "BruteForce Printing" begin
    A = [1.0 2.0; 3.0 4.0]

    sol = cutnorm(A; method=BruteForce(), print_level=1)
    @test sol.value ≈ 10.0

    sol = cutnorm(A; method=BruteForce(), print_level=3)
    @test sol.value ≈ 10.0
end
