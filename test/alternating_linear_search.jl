@testset "Alternating Linear Search Sub Solver" begin
    # All-ones: optimal s=[1,1,1], t=[1,1], value=-6
    A = -[1.0 1.0; 1.0 1.0; 1.0 1.0]
    model = BilinearModel(A)
    m, n = size(A)
    solver = AlternatingLinearSearch(model)
    stats = SolverCore.GenericExecutionStats(model)

    SolverCore.solve!(solver, model, stats; x=ones(Float64, m + n), max_iter=100)
    @test stats.objective ≈ -6.0
    @test stats.solution ≈ [1.0, 1.0, 1.0, 1.0, 1.0]
    @test stats.status ∈ (:first_order, :max_iter, :max_time)

    # Single dominant entry: optimal s=[1,0], t=[1,0], value=-5
    A = [-5.0 0.1; 0.1 0.1]
    model = BilinearModel(A)
    m, n = size(A)
    solver = AlternatingLinearSearch(model)
    stats = SolverCore.GenericExecutionStats(model)

    SolverCore.solve!(solver, model, stats; x=ones(Float64, m + n), max_iter=100)
    @test stats.objective ≈ -5.0
    @test stats.solution ≈ [1.0, 0.0, 1.0, 0.0]
    # 1×1 matrix: optimal s=[1], t=[1], value=-3
    A = reshape([-3.0], 1, 1)
    model = BilinearModel(A)
    solver = AlternatingLinearSearch(model)
    stats = SolverCore.GenericExecutionStats(model)

    SolverCore.solve!(solver, model, stats; x=ones(Float64, 2), max_iter=100)
    @test stats.objective ≈ -3.0

    # max_iter termination
    A = [1.0 1.0; 1.0 1.0]
    model = BilinearModel(A)
    solver = AlternatingLinearSearch(model)
    stats = SolverCore.GenericExecutionStats(model)

    SolverCore.solve!(solver, model, stats; x=0.5 * ones(Float64, 4), max_iter=1)
    @test stats.status == :max_iter || stats.status == :first_order
    @test stats.iter == 1 || stats.iter == 2

    A = [1.0 1.0; 1.0 1.0]
    model = BilinearModel(A)
    solver = AlternatingLinearSearch(model)
    stats = SolverCore.GenericExecutionStats(model)

    SolverCore.solve!(solver, model, stats; x=0.5 * ones(Float64, 4), max_iter=0)
    @test stats.status == :max_iter || stats.status == :first_order
    @test stats.iter == 1
    @test stats.objective ≈ 0.0

    A = -[1.0 1.0; 1.0 1.0]
    model = BilinearModel(A)
    solver = AlternatingLinearSearch(model)
    stats = SolverCore.GenericExecutionStats(model)

    SolverCore.solve!(solver, model, stats; x=ones(Float64, 4))
    @test stats.objective ≈ -4.0

    # Unbounded problem with -Inf objective
    A = [-Inf 1.0; 1.0 1.0]
    model = BilinearModel(A)
    solver = AlternatingLinearSearch(model)
    stats = SolverCore.GenericExecutionStats(model)

    SolverCore.solve!(solver, model, stats; x=0.5 * ones(Float64, 4))
    @test stats.status == :first_order
    @test stats.iter == 1
    @test stats.objective ≈ -Inf

    A = [1.0 -2.0; -2.0 1.0]
    model = BilinearModel(A)
    solver = AlternatingLinearSearch(model)
    stats = SolverCore.GenericExecutionStats(model)

    SolverCore.solve!(solver, model, stats; x=0.5 * ones(Float64, 4))
    @test stats.objective ≈ -2.0
end
