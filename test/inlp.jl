@testset "INLP Types" begin
    A = [1.0 2.0; 3.0 4.0]

    sol = INLPSolution(A)
    @test sol.dims == (2, 2)
    @test length(sol.S) == 2
    @test length(sol.T) == 2
    @test sol.termination_status == :unknown
    @test sol.solve_time ≈ 0.0

    sol2 = INLPSolution(3, 4)
    @test sol2.dims == (3, 4)
    @test length(sol2.S) == 3
    @test length(sol2.T) == 4

    sol3 = INLPSolution(Float32, 2, 3)
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

@testset "INLP Settings" begin
    settings = INLPSettings()
    @test settings.max_time == 3600.0
    @test settings.scaled == false
    @test settings.print_level == 0

    settings = INLPSettings(max_time=100.0, scaled=true, print_level=2)
    @test settings.max_time == 100.0
    @test settings.scaled == true
    @test settings.print_level == 2
end