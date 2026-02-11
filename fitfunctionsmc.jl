using Random  # for randn, clamp

# 1) Extract free parameters from your NamedTuple + mask
function flatten_free(params::NamedTuple, mask::NamedTuple)
    free_vals = Float32[]
    for k in keys(params)
        if mask[k]
            push!(free_vals, Float32(params[k]))
        end
    end
    return free_vals
end

# 2) Rebuild full NamedTuple from the flat vector
function reconstruct_full(free_vec::Vector{Float32}, params::NamedTuple, mask::NamedTuple)
    i = 1
    new = Dict{Symbol,Any}()
    for k in keys(params)
        if mask[k]
            new[k] = free_vec[i]; i += 1
        else
            new[k] = params[k]
        end
    end
    return (; new...)
end

# 3) Your χ²‐based cost (always Float32 internally)
function chi2_obj(free_vec, grad)
    fp32 = Float32.(free_vec)
    cp   = reconstruct_full(fp32, star_params, free_mask)

    # Roche‐lobe sanity check (if you still need it)
    if cp.surface_type == 3 && cp.fillout_factor_primary == 0f0
        q, a = cp.q, cp.a
        RL = a * (0.49*q^(2/3)) / (0.6*q^(2/3) + log(1+q^(1/3)))
        if cp.rpole > RL
            return 1f010
        end
    end

    # true forward model
    try
        
        stars = create_star_multiepochs(tessels, cp, tepochs)
        setup_oi!(data, stars)
        χ2 = spheroid_parametric_f(cp, tessels, data, tepochs)
        return isfinite(χ2) ? χ2 : 1f010
    catch
        return 1f010
    end
end

# 4) Simulated‐annealing + Monte Carlo
function mc_sa_cs(
    p0::Vector{Float32}, lb::Vector{Float32}, ub::Vector{Float32},
    χ2obj::Function, n_runs::Int=20,
    T0::Float32=1f04, Tf::Float32=1f0-2, α::Float32=0.995f0,
    niter::Int=10_000, step_frac::Float32=0.05f0
)
    best_fits = Vector{Vector{Float32}}(undef, n_runs)
    best_χ2   = Vector{Float32}(undef, n_runs)

    Δrange = ub .- lb
    σ      = step_frac .* Δrange

    for run in 1:n_runs
        x_cur      = copy(p0)
        cost_cur   = χ2obj(x_cur, zeros(Float32,length(p0)))
        x_best, c_best = copy(x_cur), cost_cur
        T = T0

        for i in 1:niter
            x_prop    = clamp.(x_cur .+ randn(Float32,length(p0)).*σ, lb, ub)
            cost_prop = χ2obj(x_prop, zeros(Float32,length(p0)))
            Δ         = cost_prop - cost_cur

            if Δ < 0f0 || rand() < exp(-Δ/T)
                x_cur, cost_cur = x_prop, cost_prop
                if cost_cur < c_best
                    x_best, c_best = x_cur, cost_cur
                end
            end

            T *= α
            if T < Tf
                break
            end
        end

        best_fits[run] = x_best
        best_χ2[run]   = c_best
    end

    return best_fits, best_χ2
end
