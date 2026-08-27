@testset "ILP Types" begin
    A = [1.0 2.0; 3.0 4.0]

    sol = ILPSolution(A)
    @test sol.dims == (2, 2)
    @test length(sol.S) == 2
    @test length(sol.T) == 2
    @test sol.termination_status == :unknown
    @test sol.solve_time ≈ 0.0

    sol2 = ILPSolution(3, 4)
    @test sol2.dims == (3, 4)
    @test length(sol2.S) == 3
    @test length(sol2.T) == 4

    sol3 = ILPSolution(Float32, 2, 3)
    @test sol3.dims == (2, 3)
    @test eltype(sol3.S) == Float32

    sol.value = 42.0
    sol.runtime = 1.5
    sol.solve_time = 1.2
    sol.termination_status = :optimal
    CutNorm.reset!(sol)
    @test sol.value ≈ 0.0
    @test sol.runtime ≈ 0.0
    @test sol.solve_time ≈ 0.0
    @test sol.termination_status == :unknown
end

@testset "ILP Settings" begin
    settings = ILPSettings()
    @test settings.max_time == 3600.0
    @test settings.scaled == false
    @test settings.print_level == 0

    settings = ILPSettings(max_time=100.0, scaled=true, print_level=2)
    @test settings.max_time == 100.0
    @test settings.scaled == true
    @test settings.print_level == 2
end

@testset "ILP big-M" begin
    # 5x5 sanity matrix: sum_pos = 15, -sum_neg = 10 -> M = 2*15 = 30
    A = [1.0 1 1 1 1;
        1 1 1 1 1;
        1 1 1 1 1;
        -1 -1 -1 -1 -1;
        -1 -1 -1 -1 -1]
    @test CutNorm.compute_bigM(A) ≈ 30.0

    # all positive: sum_pos = 6, sum_neg = 0 -> M = 12
    @test CutNorm.compute_bigM([1.0 1.0; 1.0 1.0; 1.0 1.0]) ≈ 12.0

    # all negative: -sum_neg = 18 -> M = 36
    @test CutNorm.compute_bigM([-3.0 -4.0; -5.0 -6.0]) ≈ 36.0

    # zero matrix: M = 0 (uses init to avoid reducing over an empty collection)
    @test CutNorm.compute_bigM(zeros(3, 3)) ≈ 0.0

    # type is preserved
    @test CutNorm.compute_bigM(Float32[1 -2; -3 4]) isa Float32
end
