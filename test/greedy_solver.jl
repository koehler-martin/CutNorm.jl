@testset "Greedy Sub Solver" begin
    # All-ones: optimal s=[1,1,1], t=[1,1], value=-6
    A = -[1.0 1.0; 1.0 1.0; 1.0 1.0]
    model = BilinearModel(A)
    m, n = size(A)
    solver = GreedySolver(model)
    stats = SolverCore.GenericExecutionStats(model)

    SolverCore.solve!(solver, model, stats; x=ones(Float64, m + n), max_iter=100)
    @test stats.objective ≈ -6.0
    @test stats.solution ≈ [1.0, 1.0, 1.0, 1.0, 1.0]
    @test stats.status ∈ (:first_order, :max_iter, :max_time)

    # Single dominant entry: optimal s=[1,0], t=[1,0], value=-5
    A = [-5.0 0.1; 0.1 0.1]
    model = BilinearModel(A)
    m, n = size(A)
    solver = GreedySolver(model)
    stats = SolverCore.GenericExecutionStats(model)

    SolverCore.solve!(solver, model, stats; x=ones(Float64, m + n), max_iter=100)
    @test stats.objective ≈ -5.0
    @test stats.solution ≈ [1.0, 0.0, 1.0, 0.0]

    # 1×1 matrix: optimal s=[1], t=[1], value=-3
    A = reshape([-3.0], 1, 1)
    model = BilinearModel(A)
    solver = GreedySolver(model)
    stats = SolverCore.GenericExecutionStats(model)

    SolverCore.solve!(solver, model, stats; x=ones(Float64, 2), max_iter=100)
    @test stats.objective ≈ -3.0

    # max_iter termination
    A = [1.0 1.0; 1.0 1.0]
    model = BilinearModel(A)
    solver = GreedySolver(model)
    stats = SolverCore.GenericExecutionStats(model)

    SolverCore.solve!(solver, model, stats; x=0.5 * ones(Float64, 4), max_iter=1)
    @test stats.status == :max_iter || stats.status == :first_order
    @test stats.iter >= 1

    # Negative matrix
    A = -[1.0 1.0; 1.0 1.0]
    model = BilinearModel(A)
    solver = GreedySolver(model)
    stats = SolverCore.GenericExecutionStats(model)

    SolverCore.solve!(solver, model, stats; x=ones(Float64, 4))
    @test stats.objective ≈ -4.0

    # Off-diagonal dominant
    A = [1.0 -2.0; -2.0 1.0]
    model = BilinearModel(A)
    solver = GreedySolver(model)
    stats = SolverCore.GenericExecutionStats(model)

    SolverCore.solve!(solver, model, stats; x=0.5 * ones(Float64, 4))
    @test stats.objective ≈ -2.0
end

@testset "Greedy Multistart Integration" begin
    # All-ones: cut norm = 6
    A = [1.0 1.0;
        1.0 1.0;
        1.0 1.0]
    expected_S = [1.0, 1.0, 1.0]
    expected_T = [1.0, 1.0]

    solver = MultistartSignedSolver(A, GreedySolver)
    sol = MultistartSignedSolution(A)

    solve!(solver, sol)
    @test sol.value ≈ 6.0
    @test sol.S ≈ expected_S
    @test sol.T ≈ expected_T
    @test sol.restarts >= 1
    @test sol.improvements >= 1

    # Dominant entry: cut norm = 10
    A = -[-10.0 1.0;
        1.0 1.0;
        1.0 1.0]
    expected_S = [1.0, 0.0, 0.0]
    expected_T = [1.0, 0.0]

    solver = MultistartSignedSolver(A, GreedySolver)
    sol = MultistartSignedSolution(A)

    solve!(solver, sol)
    @test sol.value ≈ 10.0
    @test sol.S ≈ expected_S
    @test sol.T ≈ expected_T
    @test abs(dot(sol.S, A, sol.T)) ≈ sol.value

    # cutnorm dispatch
    solution = cutnorm(A; method=MultistartAugmented{GreedySolver}())
    @test solution.value ≈ 10.0
end
