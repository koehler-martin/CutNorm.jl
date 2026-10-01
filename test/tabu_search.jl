# Exact cut norm by enumeration, to compare the tabu search with
tabu_exact(A) = cutnorm(Matrix(A); method=BruteForce()).value

is_indicator(x) = all(xi -> xi == 0 || xi == 1, x)

# Run f with stdout redirected to a file and return its result and the printed output
function tabu_capture(f)
    path, io = mktemp()
    result = redirect_stdout(f, io)
    close(io)
    output = read(path, String)
    rm(path)
    return result, output
end

# Rows of the live log, split into columns: the lines between its header and the next rule
function tabu_log_rows(output)
    lines = split(output, '\n')
    first_row = findfirst(l -> occursin("Thread    Phase", l), lines) + 2
    last_row = findnext(l -> startswith(l, "-----"), lines, first_row) - 1
    return [split(l) for l in lines[first_row:last_row]]
end

@testset "TabuSearch Settings" begin
    s = TabuSearchSettings()
    @test s.max_iter == typemax(Int)
    @test s.max_time == 10.0
    @test s.target == Inf
    @test s.ntasks == Threads.nthreads()
    @test s.S0 === nothing
    @test s.T0 === nothing
    @test !s.scaled
    @test s.print_level == 0

    A = [1.0 2.0; 3.0 4.0]
    solver = TabuSearchSolver(A; max_time=5.0, seed=42, S0=[1.0, 0.0], T0=[0, 1])
    @test solver.settings.max_time == 5.0
    @test solver.settings.seed === UInt64(42)
    @test solver.settings.S0 == [true, false]
    @test solver.settings.T0 == [false, true]

    @test_throws ArgumentError TabuSearchSolver(A; restarts=3)
    @test_throws ArgumentError solve!(TabuSearchSolver(A); max_restarts=3)
end

@testset "TabuSearch cutnorm" begin
    A = [1.0 1 1 1 1;
        1 1 1 1 1;
        1 1 1 1 1;
        -1 -1 -1 -1 -1;
        -1 -1 -1 -1 -1]

    sol = cutnorm(A; method=TabuSearch(), max_iter=1000)
    @test sol isa TabuSearchSolution{Float64}
    @test sol.value ≈ 15.0
    @test sol.S ≈ [1.0, 1.0, 1.0, 0.0, 0.0]
    @test sol.T ≈ [1.0, 1.0, 1.0, 1.0, 1.0]
    @test sol.termination_status == :max_iter

    # Keywords are forwarded to the settings
    sol = cutnorm(A; method=TabuSearch(), max_iter=1000, ntasks=2, scaled=true)
    @test sol.value ≈ 15.0 / 25
    @test sol.iterations == 2 * 1000
    @test_throws ArgumentError cutnorm(A; method=TabuSearch(), max_restarts=10)

    # Same result as the solver API with the same seed
    B = randn(Xoshiro(14), 20, 15)
    a = cutnorm(B; method=TabuSearch(), max_iter=3000, ntasks=2, seed=5)
    b = solve!(TabuSearchSolver(B; max_iter=3000, ntasks=2, seed=5))
    @test a.value == b.value
    @test a.S == b.S && a.T == b.T

    # Agrees with the exact method, also for sparse input
    C = randn(Xoshiro(15), 6, 7)
    exact = cutnorm(C; method=BruteForce()).value
    @test cutnorm(C; method=TabuSearch(), max_iter=20_000).value ≈ exact
    @test cutnorm(sparse(C); method=TabuSearch(), max_iter=20_000).value ≈ exact

    sol = cutnorm(B; method=TabuSearch(), max_time=0.2, ntasks=1)
    @test sol.termination_status == :max_time
end

@testset "TabuSearch Types" begin
    A = [1.0 2.0 3.0; 4.0 5.0 6.0]

    solver = TabuSearchSolver(A; ntasks=3)
    @test solver isa CutNorm.AbstractSolver{Float64}
    @test solver.A == A
    @test solver.A !== A
    @test solver.At == permutedims(A)
    @test length(solver.workspaces) == 3
    ws = solver.workspaces[1]
    @test length(ws.s) == 2 && length(ws.t) == 3
    @test length(ws.bs) == 2 && length(ws.bt) == 3
    @test length(ws.tabu_until) == 5 && length(ws.perm) == 5

    sol = TabuSearchSolution(A)
    @test sol isa TabuSearchSolution{Float64}
    @test sol.dims == (2, 3)
    @test length(sol.S) == 2
    @test length(sol.T) == 3
    @test sol.termination_status == :unknown

    @test TabuSearchSolution(3, 4).dims == (3, 4)
    @test eltype(TabuSearchSolution(Float32, 2, 3).S) == Float32

    sol.value = 42.0
    sol.sign = Int8(-1)
    sol.iterations = 7
    sol.S .= 1
    CutNorm.reset!(sol)
    @test sol.value == 0.0
    @test sol.sign == 0
    @test sol.iterations == 0
    @test all(iszero, sol.S)
    @test sol.termination_status == :unknown
end

@testset "TabuSearch Known Values" begin
    # 5x5 sanity check matrix from the main test suite
    A = [1.0 1 1 1 1;
        1 1 1 1 1;
        1 1 1 1 1;
        -1 -1 -1 -1 -1;
        -1 -1 -1 -1 -1]

    sol = solve!(TabuSearchSolver(A; max_iter=1000, ntasks=2))
    @test sol.value ≈ 15.0
    @test sol.S ≈ [1.0, 1.0, 1.0, 0.0, 0.0]
    @test sol.T ≈ [1.0, 1.0, 1.0, 1.0, 1.0]
    @test sol.sign == 1
    @test sol.termination_status == :max_iter

    sol = solve!(TabuSearchSolver(A; max_iter=1000, scaled=true))
    @test sol.value ≈ 15.0 / 25

    # The matrices of the BruteForce tests: (A, cut norm, sign of the optimal sum)
    cases = [
        ([1.0 1.0; 1.0 1.0; 1.0 1.0], 6.0, 1),
        ([10.0 -1.0; -1.0 -1.0; -1.0 -1.0], 10.0, 1),
        (reshape([7.0], 1, 1), 7.0, 1),
        (reshape([-7.0], 1, 1), 7.0, -1),
        ([-3.0 -4.0; -5.0 -6.0], 18.0, -1),
        ([1.0 -2.0; -2.0 1.0], 2.0, -1),
    ]
    for (A, value, sign) in cases
        sol = solve!(TabuSearchSolver(A; max_iter=1000, ntasks=2))
        @test sol.value ≈ value
        @test sol.sign == sign
        @test abs(dot(sol.S, A, sol.T)) ≈ sol.value
    end

    # Zero matrix
    sol = solve!(TabuSearchSolver(zeros(3, 3); max_iter=1000))
    @test sol.value == 0.0
    @test sol.sign == 1
end

@testset "TabuSearch vs BruteForce" begin
    rng = Xoshiro(1)
    for (m, n) in [(1, 5), (5, 1), (2, 2), (3, 4), (5, 5), (6, 8), (8, 6), (8, 8)], ntasks in (1, 3)
        A = randn(rng, m, n)
        sol = solve!(TabuSearchSolver(A; max_iter=20_000, ntasks=ntasks, seed=m + n))
        @test sol.value ≈ tabu_exact(A)
        @test sol.value ≈ abs(dot(sol.S, A, sol.T))
        @test sol.sign == sign(dot(sol.S, A, sol.T))
        @test is_indicator(sol.S) && is_indicator(sol.T)
        @test sol.iterations == ntasks * 20_000
        @test 1 <= sol.best_task <= ntasks
        @test 0 <= sol.best_iteration <= 20_000
    end
end

@testset "TabuSearch Sign" begin
    # The cut norm 12 is attained only with the negative sign
    A = -ones(3, 4)

    # A single search that never restarts keeps the + sign and cannot find it
    sol = solve!(TabuSearchSolver(A; max_iter=5000, ntasks=1, max_perturbations=typemax(Int)))
    @test sol.restarts == 0
    @test sol.value == 0.0

    # The second search starts with the - sign
    sol = solve!(TabuSearchSolver(A; max_iter=5000, ntasks=2, max_perturbations=typemax(Int)))
    @test sol.value ≈ 12.0
    @test sol.sign == -1
    @test sol.best_task == 2

    # A restart switches the sign
    sol = solve!(TabuSearchSolver(A; max_iter=5000, ntasks=1))
    @test sol.restarts > 0
    @test sol.value ≈ 12.0
    @test sol.sign == -1
end

@testset "TabuSearch Diversification" begin
    A = randn(Xoshiro(2), 8, 7)
    exact = tabu_exact(A)

    # A short stall forces many perturbations and restarts
    sol = solve!(TabuSearchSolver(A; max_iter=20_000, ntasks=2, stall=20))
    @test sol.perturbations > 0
    @test sol.restarts > 0
    @test sol.value ≈ exact

    # Without perturbations every stall ends in a restart
    sol = solve!(TabuSearchSolver(A; max_iter=20_000, ntasks=2, stall=20, max_perturbations=0))
    @test sol.perturbations == 0
    @test sol.restarts > 0
    @test sol.value ≈ exact

    # Tenure and perturbation size at their extremes
    sol = solve!(TabuSearchSolver(A; max_iter=20_000, ntasks=2, stall=20, tenure=1, tenure_rand=0, perturb_frac=1.0))
    @test sol.value ≈ exact
    sol = solve!(TabuSearchSolver(A; max_iter=20_000, ntasks=2, stall=20, tenure=10, tenure_rand=100, perturb_frac=0.0))
    @test sol.value ≈ exact
end

@testset "TabuSearch Tabu List" begin
    # Without diversification the tabu list is never cleared, so after the last flip
    # (iteration 200) its bit stays tabu for tenure + rand(0:tenure_rand) iterations
    A = randn(Xoshiro(12), 10, 12)
    solver = TabuSearchSolver(A; ntasks=1, max_iter=200, stall=10^9, tenure=3, tenure_rand=2)
    solve!(solver)
    ws = solver.workspaces[1]
    @test 200 + 3 <= maximum(ws.tabu_until) <= 200 + 3 + 2
    @test count(>=(200), ws.tabu_until) <= 3 + 2 + 1   # only the last few flips are still tabu
end

@testset "TabuSearch Perturbation" begin
    # Started at the optimum, the elite cannot improve, so the first stall ends in a
    # perturbation in iteration stall + 1, which flips round(perturb_frac * N) bits of it
    A = randn(Xoshiro(13), 6, 7)
    bf = cutnorm(A; method=BruteForce())
    solver = TabuSearchSolver(A; ntasks=1, S0=bf.S, T0=bf.T, stall=30, max_iter=31, perturb_frac=0.3)
    sol = solve!(solver)
    @test sol.perturbations == 1
    @test sol.restarts == 0
    @test sol.value ≈ bf.value
    ws = solver.workspaces[1]
    @test ws.es == bf.S && ws.et == bf.T
    @test count(ws.s .!= ws.es) + count(ws.t .!= ws.et) == round(Int, 0.3 * 13)
end

@testset "TabuSearch Termination" begin
    A = [1.0 1 1 1 1;
        1 1 1 1 1;
        1 1 1 1 1;
        -1 -1 -1 -1 -1;
        -1 -1 -1 -1 -1]

    # max_iter limits every search
    sol = solve!(TabuSearchSolver(A; max_iter=500, ntasks=3))
    @test sol.termination_status == :max_iter
    @test sol.iterations == 3 * 500

    # max_iter = 0 only evaluates the initial points
    sol = solve!(TabuSearchSolver(A; max_iter=0, ntasks=2))
    @test sol.termination_status == :max_iter
    @test sol.iterations == 0
    @test sol.value ≈ abs(dot(sol.S, A, sol.T))

    # Reaching the target stops all searches long before the time limit
    sol = solve!(TabuSearchSolver(A; target=15.0, ntasks=4, max_time=10.0))
    @test sol.termination_status == :target
    @test sol.value ≈ 15.0
    @test sol.runtime < 5.0

    # With scaled = true, the target is compared with the scaled value
    sol = solve!(TabuSearchSolver(A; target=0.6, scaled=true, max_time=10.0))
    @test sol.termination_status == :target
    @test sol.value ≈ 0.6

    # An initial guess that already reaches the target
    sol = solve!(TabuSearchSolver(A; target=15.0, ntasks=1, S0=[1, 1, 1, 0, 0], T0=ones(5)))
    @test sol.termination_status == :target
    @test sol.iterations == 0

    B = randn(Xoshiro(3), 40, 30)
    sol = solve!(TabuSearchSolver(B; max_time=0.5, ntasks=1))
    @test sol.termination_status == :max_time
    @test sol.runtime >= 0.5
    @test sol.time_to_best <= sol.runtime
    @test sol.value ≈ abs(dot(sol.S, B, sol.T))
end

@testset "TabuSearch Reproducibility" begin
    A = randn(Xoshiro(4), 30, 40)

    solver = TabuSearchSolver(A; max_iter=3000, ntasks=3, seed=7, stall=50)
    a = solve!(solver)
    b = solve!(solver)
    @test a.value == b.value
    @test a.S == b.S && a.T == b.T
    @test a.best_task == b.best_task && a.best_iteration == b.best_iteration
    @test a.perturbations == b.perturbations && a.restarts == b.restarts

    c = solve!(TabuSearchSolver(A; max_iter=3000, ntasks=3, seed=7, stall=50))
    @test c.value == a.value
    @test c.S == a.S && c.T == a.T
end

@testset "TabuSearch Initial Guess" begin
    A = randn(Xoshiro(5), 6, 7)
    bf = cutnorm(A; method=BruteForce())

    # With max_iter = 0 the initial guess is returned unchanged
    sol = solve!(TabuSearchSolver(A; max_iter=0, ntasks=1, S0=bf.S, T0=bf.T))
    @test sol.value ≈ bf.value
    @test sol.S == bf.S && sol.T == bf.T

    # The first search takes the sign of the initial guess, so this works for -A too
    sol = solve!(TabuSearchSolver(-A; max_iter=0, ntasks=1, S0=bf.S, T0=bf.T))
    @test sol.value ≈ bf.value
    @test sol.sign == -sign(dot(bf.S, A, bf.T))

    # Only S0 given: T starts at random
    sol = solve!(TabuSearchSolver(A; max_iter=20_000, S0=bf.S))
    @test sol.value ≈ bf.value

    @test_throws DimensionMismatch solve!(TabuSearchSolver(A; S0=ones(5)))
    @test_throws DimensionMismatch solve!(TabuSearchSolver(A; T0=ones(8)))
end

@testset "TabuSearch Direct API" begin
    A = randn(Xoshiro(6), 5, 6)
    exact = tabu_exact(A)

    solver = TabuSearchSolver(A; max_iter=5000, ntasks=1)
    sol = TabuSearchSolution(A)
    @test solve!(solver, sol) === sol
    @test sol.value ≈ exact
    @test sol.runtime > 0.0

    # Settings persist between calls; more searches add workspaces
    solve!(solver, sol; ntasks=3)
    @test length(solver.workspaces) == 3
    @test solver.settings.max_iter == 5000
    @test sol.iterations == 3 * 5000
    @test sol.value ≈ exact

    # The solution is reset before it is reused
    solve!(solver, sol; max_iter=0)
    @test sol.iterations == 0
    @test sol.perturbations == 0
end

@testset "TabuSearch Incremental Updates" begin
    # Without refreshes and diversification, u = A*t and v = A'*s are only ever updated
    # incrementally, one row or column per flip
    A = randn(Xoshiro(7), 12, 9)
    solver = TabuSearchSolver(A; max_iter=2000, ntasks=2, stall=10^9, refresh=10^9)
    sol = solve!(solver)
    @test sol.perturbations == 0 && sol.restarts == 0
    for ws in solver.workspaces
        @test is_indicator(ws.s) && is_indicator(ws.t)
        @test is_indicator(ws.bs) && is_indicator(ws.bt)
        @test ws.v ≈ A' * ws.s
    end
    # solve! overwrites u of the best search when it recomputes the value
    other = solver.workspaces[3-sol.best_task]
    @test other.u ≈ A * other.t
end

@testset "TabuSearch Matrix Types" begin
    # Sparse matrices stay sparse and give the same result as their dense copy
    A = sprandn(Xoshiro(8), 60, 50, 0.1)
    solver = TabuSearchSolver(A; max_iter=5000, ntasks=2, seed=1)
    @test solver.A isa SparseMatrixCSC{Float64,Int}
    @test solver.At isa SparseMatrixCSC{Float64,Int}
    @test solver.At == permutedims(A)
    sp = solve!(solver)
    de = solve!(TabuSearchSolver(Matrix(A); max_iter=5000, ntasks=2, seed=1))
    @test sp.value ≈ de.value
    @test sp.value ≈ abs(dot(sp.S, A, sp.T))

    B = sprandn(Xoshiro(9), 6, 7, 0.5)
    @test solve!(TabuSearchSolver(B; max_iter=20_000)).value ≈ tabu_exact(B)

    # Wrappers are copied into dense matrices
    C = randn(Xoshiro(10), 6, 5)
    for X in (C', transpose(C), view(C, :, :))
        solver = TabuSearchSolver(X; max_iter=20_000)
        @test solver.A isa Matrix{Float64}
        @test solve!(solver).value ≈ tabu_exact(X)
    end
    for X in (Symmetric(C[1:5, :]), Diagonal(C[1:5, 1]))
        @test solve!(TabuSearchSolver(X; max_iter=20_000)).value ≈ tabu_exact(X)
    end

    C32 = Float32.(C)
    sol = solve!(TabuSearchSolver(C32; max_iter=20_000))
    @test sol isa TabuSearchSolution{Float32}
    @test sol.value ≈ tabu_exact(C32)
end

@testset "TabuSearch Printing" begin
    A = randn(Xoshiro(11), 10, 12)

    sol, out = tabu_capture(() -> solve!(TabuSearchSolver(A; max_iter=2000)))
    @test isempty(out)

    for pl in 1:3
        sol, out = tabu_capture(() -> solve!(TabuSearchSolver(A; max_iter=2000, ntasks=2, stall=50, print_level=pl)))
        @test occursin("Tabu Search", out)
        @test occursin("Terminated:        max_iter", out)

        rows = tabu_log_rows(out)
        best = [parse(Float64, r[3]) for r in rows]
        improvements = [parse(Int, r[4]) for r in rows]
        @test issorted(best) && issorted(improvements)
        @test count(r -> length(r) == 8 && r[8] == "*", rows) == sol.improvements
        @test improvements[end] == sol.improvements
        @test best[end] ≈ sol.value rtol = 1e-4
        if pl == 1
            @test length(rows) == sol.improvements
        elseif pl == 3
            @test length(rows) == sol.perturbations + sol.restarts + 2  # one per phase
        end
    end
end
