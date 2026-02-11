using Random          # for rand()
using NLopt
function vectorized_prior_transform(usamples::Matrix{Float64})::Matrix{Float64}
    n_samples = size(usamples, 1)
    psamples = Matrix{Float64}(undef, n_samples, 3)

    @inbounds for i in 1:n_samples
        u = clamp.(usamples[i, :], 1e-6, 1 - 1e-6)
        M2 = quantile(prior_M2, Float32(u[1]))
        i_ = quantile(prior_i,  Float32(u[2]))
        M1 = quantile(prior_M1, Float32(u[3]))
        psamples[i, :] .= (Float64(M2), Float64(i_), Float64(M1))
    end

    return psamples  # shape (n_samples, 3)
end

function vectorized_loglike(ps::Matrix{Float64})::Vector{Float64}
    n_samples = size(ps, 1)
    loglikes = Vector{Float64}(undef, n_samples)

    @inbounds for j in 1:n_samples
        M2 = Float32(ps[j, 1])
        i  = Float32(ps[j, 2])
        M1 = Float32(ps[j, 3])

        fm_pred = (M2 * sin(deg2rad(i)))^3 / (M1 + M2)^2

        if !isfinite(fm_pred) || fm_pred <= 0
            loglikes[j] = -1e5  # soft penalty
        else
            chi2 = ((fm_pred - fm_obs) / sigma_fm)^2
            loglikes[j] = -0.5 * chi2
        end
    end

    return loglikes
end

# --- Vectorized prior transform ---
function vectorized_prior_transform(usamples::Matrix{Float64})::Matrix{Float64}
    n_samples = size(usamples, 1)
    psamples = Matrix{Float64}(undef, n_samples, 3)

    @inbounds for i in 1:n_samples
        u = clamp.(usamples[i, :], 1e-6, 1 - 1e-6)
        M2 = quantile(prior_M2, Float32(u[1]))
        i_ = quantile(prior_i,  Float32(u[2]))
        M1 = quantile(prior_M1, Float32(u[3]))
        psamples[i, :] .= (Float64(M2), Float64(i_), Float64(M1))
    end

    return psamples  # shape (n_samples, 3)
end



function a_mas_from_masses(M1, M2, P_days, d_pc)
    G = 39.478 # AU^3 / (Msun * yr^2)
    P_yr = P_days / 365.25
    a_AU = ((G * (M1 + M2) * P_yr^2) / (4π^2))^(1/3)
    a_mas = (a_AU / d_pc) * 1000
    return a_mas
end



function flatten_free(params::NamedTuple, mask::NamedTuple)
    free_vals = Float32[]
    for key in keys(params)
        if mask[key]
            push!(free_vals, Float32(params[key]))
        end
    end
    return free_vals
end


function reconstruct_full(free_vec::Vector{Float32}, params::NamedTuple, mask::NamedTuple;)# alignments = Dict{Symbol,Symbol}())
    new_params = Dict{Symbol,Any}()
    i = 1
    for key in keys(params)
        if mask[key]
            new_params[key] = free_vec[i]
            i += 1
        else
            new_params[key] = params[key]
        end
    end
    # Enforce alignment constraints:
  #  for (k_target, k_source) in alignments
        #new_params[k_target] = new_params[k_source]
   # end
    return (; new_params...)
end



using Random          # for rand()
using NLopt


function resample_data(data_input;weights=weights)
    data_out = deepcopy(data_input[1]);
    # V2 bootstrap
    if weights[1]>0
        indx_resampling = Int.(ceil.(data_input[1].nv2*rand(data_input[1].nv2)));
        data_out.v2 = data_input[1].v2[indx_resampling];
        data_out.v2_err = data_input[1].v2_err[indx_resampling];
        data_out.v2_baseline = data_input[1].v2_baseline[indx_resampling];
        data_out.nv2 = length(data_input[1].v2);
        data_out.v2_mjd  = data_input[1].v2_mjd[indx_resampling];
        data_out.v2_lam  = data_input[1].v2_lam[indx_resampling];
        data_out.v2_dlam = data_input[1].v2_dlam[indx_resampling];
        data_out.v2_flag = data_input[1].v2_flag[indx_resampling];
        data_out.v2_sta_index = data_input[1].v2_sta_index[:,indx_resampling];
        data_out.indx_v2= data_input[1].indx_v2[indx_resampling];
    end
    # T3 bootstrap (amp + phi together)
    if weights[2]>0 || weights[3]>0
        indx_resampling = Int.(ceil.(data_input[1].nt3phi*rand(data_input[1].nt3phi))); # needs updating if nt3amp =/= nt3phi;
        data_out.t3amp = data_input[1].t3amp[indx_resampling];
        data_out.t3amp_err = data_input[1].t3amp_err[indx_resampling];
        data_out.nt3amp = data_input[1].nt3amp;
        data_out.t3phi = data_input[1].t3phi[indx_resampling];
        data_out.t3phi_err = data_input[1].t3phi_err[indx_resampling];
        data_out.nt3phi = data_input[1].nt3phi;
        data_out.t3_baseline = data_input[1].t3_baseline[indx_resampling];
        data_out.t3_mjd  = data_input[1].t3_mjd[indx_resampling];
        data_out.t3_lam  = data_input[1].t3_lam[indx_resampling];
        data_out.t3_dlam = data_input[1].t3_dlam[indx_resampling];
        data_out.t3_flag = data_input[1].t3_flag[indx_resampling];
        data_out.t3_sta_index = data_input[1].t3_sta_index[:,indx_resampling];
        data_out.indx_t3_1= data_input[1].indx_t3_1[indx_resampling];
        data_out.indx_t3_2= data_input[1].indx_t3_2[indx_resampling];
        data_out.indx_t3_3= data_input[1].indx_t3_3[indx_resampling];
       
    end
    return data_out
end

# 1) Tighter tolerances and more evaluations
function make_optimizer(p0::Vector{Float32}, lb::Vector, ub::Vector; fitter=:LN_COBYLA)
    opt = Opt(fitter, length(p0))
    min_objective!(opt, chi2_obj)
    xtol_rel!(opt, 1e-6)    # tighter on x
    ftol_rel!(opt, 1e-8)    # also stop when χ² stops changing
    maxeval!(opt, 50_000)   # more chances to climb out of p0
    return opt
end

# 2) Jitter the initial guess on each bootstrap
function bootstrap_nlopt(
    data_in, star_params, free_mask, lbounds, ubounds;
    nbootstraps::Int = 500,
    fitter = :LN_COBYLA,
    weights = [1.0,1.0,1.0],
    verbose = false
)
    p0   = flatten_free(star_params, free_mask)
    span = ubounds .- lbounds
    nfree= length(p0)
    allp = zeros(Float32, nfree, nbootstraps)

    global data = data_in   # initial data for chi2_obj

    for k in 1:nbootstraps
        # resample the data
        global data = resample_data(data_in; weights=weights)

        # make a fresh optimizer with tighter settings
        opt = make_optimizer(p0, lbounds, ubounds; fitter=fitter)

        # perturb the start by up to ±2% of the allowed range
        p0j = p0 .+ 0.02f0 .* span .* randn(Float32, nfree)

        # run NLopt quietly if you like
        _, pbest, _ = optimize(opt, p0j)

        allp[:, k] = Float32.(pbest)

        if verbose && k % ceil(nbootstraps/10) == 0
            println("Bootstrap $k/$nbootstraps, sample pbest = ", pbest)
        end
    end

    return allp
end

function bootstrap_mc_crs(
    data0, star_params, free_mask, lbounds, ubounds;
    nbootstraps = 500
)
    p0   = flatten_free(star_params, free_mask);
    nfree= length(p0);
    allp = zeros(Float32, nfree, nbootstraps);

    for k in 1:nbootstraps
        # 1) Monte Carlo resample
        global data = resample_data_mc(data0);

        # 2) make a global optimizer
        opt = Opt(:GN_CRS2_LM, nfree);
        min_objective!(opt, chi2_obj);
        lower_bounds!(opt, lbounds);
        upper_bounds!(opt, ubounds);
        xtol_rel!(opt,1e-6); maxeval!(opt,50_000);

        # 3) run it from a jittered start
        p0j = p0 .+ 0.02f0 * (ubounds .- lbounds) .* randn(Float32, nfree);
        _, pbest, _ = optimize(opt, p0j);

        allp[:, k] = Float32.(pbest);
    end

    return allp
end

using Distributions


function resample_data_mc(data_in::OIdata{Float32})
    # deep‐copy the entire struct
    data_out = deepcopy(data_in);

    # 1) V²: sample from N(v2, v2_err²)
    data_out.v2   = [ rand(Normal(m,σ)) for (m,σ) in zip(data_in.v2,   data_in.v2_err) ];
    # leave v2_err alone

    # 2) T3 amp: same idea
    data_out.t3amp = [ rand(Normal(m,σ)) for (m,σ) in zip(data_in.t3amp, data_in.t3amp_err) ];

    # 3) T3 phase: add wrapped Gaussian noise
    data_out.t3phi = mod.(data_in.t3phi .+ randn(length(data_in.t3phi)) .* data_in.t3phi_err, 360f0);

    return data_out
end


# =======================
# Objective Function for NLopt (Roche Model)
# =======================
function chi2_obj(free_vec, grad)
    free_vec32 = Float32.(free_vec)
    current_params = reconstruct_full(free_vec32, star_params, free_mask)
    
    # If fillout_factor_primary > 0, skip the Roche-lobe check.
    if current_params.surface_type == 3
        if current_params.fillout_factor_primary > 0
            nothing
        else
            q_val = current_params.q
            a_val = current_params.a
            R_L = a_val * (0.49 * q_val^(2/3)) / (0.6 * q_val^(2/3) + log(1 + q_val^(1/3)))
            if current_params.rpole > R_L
                return 1e10
            end
        end
    end    
    local chi2 = 1e6
    try
        stars = create_star_multiepochs(tessels, current_params, tepochs)
        setup_oi!(data, stars)
        chi2 = spheroid_parametric_f(current_params, tessels, data, tepochs)
    catch e
        println("Error in forward model: ", e)
        chi2 = 1e10
    end
    if !isfinite(chi2) || isnan(chi2)
        chi2 = 1e10
    end
    return chi2
end

using Random

# 1) data‑compression function
#    randomly subsample V² & T³ by fraction `frac`
function compress_data(data_full::OIdata{Float32}, frac::Float32)
    df = deepcopy(data_full);
    # V²
    n2 = df.nv2;
    k2 = max(1, Int(round(frac * n2)));
    idx2 = randperm(n2)[1:k2];
    df.v2           = df.v2[idx2];
    df.v2_err       = df.v2_err[idx2];
    df.v2_baseline  = df.v2_baseline[idx2];
    df.v2_mjd       = df.v2_mjd[idx2];
    df.v2_lam       = df.v2_lam[idx2];
    df.v2_dlam      = df.v2_dlam[idx2];
    df.v2_flag      = df.v2_flag[idx2];
    df.v2_sta_index = df.v2_sta_index[:, idx2];
    df.indx_v2      = df.indx_v2[idx2];
    df.nv2          = k2;

    # T³ (amp + phi)
    n3 = df.nt3phi;
    k3 = max(1, Int(round(frac * n3)));
    idx3 = randperm(n3)[1:k3];
    df.t3amp        = df.t3amp[idx3];
    df.t3amp_err    = df.t3amp_err[idx3];
    df.t3phi        = df.t3phi[idx3];
    df.t3phi_err    = df.t3phi_err[idx3];
    df.t3_baseline  = df.t3_baseline[idx3];
    df.t3_mjd       = df.t3_mjd[idx3];
    df.t3_lam       = df.t3_lam[idx3];
    df.t3_dlam      = df.t3_dlam[idx3];
    df.t3_flag      = df.t3_flag[idx3];
    df.t3_sta_index = df.t3_sta_index[:, idx3];
    df.indx_t3_1    = df.indx_t3_1[idx3];
    df.indx_t3_2    = df.indx_t3_2[idx3];
    df.indx_t3_3    = df.indx_t3_3[idx3];
    df.nt3amp       = k3;
    df.nt3phi       = k3;

    return df
end;


# 2) wrap your χ² to use compressed data each eval
function make_chi2_compressed(
    data_full, frac, chi2obj,
    star_params, free_mask, tessels, tepochs
)
    function χ2_compressed(free_vec, grad)
        # grab a fresh random subset
        global data = compress_data(data_full, frac);
        return chi2obj(free_vec, grad);
    end;
    return χ2_compressed;
end;

# 3) Monte‑Carlo of Simulated Annealing runs
function mc_sa_cs(
    star_params, free_mask, tessels, tepochs, data_full;
    lbounds, ubounds,
    chi2obj,
    n_runs::Int = 20,
    frac::Float32 = 0.2f0,     # compress to 20% of points
    # annealing schedule
    T0::Float32 = 1f4,
    Tf::Float32 = 1f-2,
    α::Float32  = 0.995f0,
    niter::Int  = 10_000,
    step_frac::Float32 = 0.05f0
)
    p0 = flatten_free(star_params, free_mask);
    lb = Float32.(lbounds);
    ub = Float32.(ubounds);

    best_fits = Vector{Vector{Float32}}(undef, n_runs);
    best_χ2  = Vector{Float32}(undef, n_runs);

    for run in 1:n_runs
        # build a fresh compressed‐χ²
        χ2c = make_chi2_compressed(data_full, frac, chi2obj,
                                   star_params, free_mask,
                                   tessels, tepochs);

        # simulated annealing on this compressed data
        x_cur = copy(p0);
        cost_cur = χ2c(x_cur, zeros(Float32,length(x_cur)));
        x_best, cost_best = copy(x_cur), cost_cur;

        T = T0;
        Δrange = ub .- lb;
        σ = step_frac .* Δrange;

        for i in 1:niter
            x_prop = clamp.(x_cur .+ randn(Float32, length(p0)).*σ, lb, ub);
            cost_prop = χ2c(x_prop, zeros(Float32,length(p0)));
            Δ = cost_prop - cost_cur;
            if Δ < 0f0 || rand() < exp(-Δ/T);
                x_cur, cost_cur = x_prop, cost_prop;
                if cost_cur < cost_best
                    x_best, cost_best = x_cur, cost_cur;
                end
            end
            T *= α
            if T < Tf
                break;
            end
        end

        best_fits[run] = x_best;
        best_χ2[run]  = cost_best;
    end

    return best_fits, best_χ2;
end;
#------------------------------
using Plots, StatsPlots, LaTeXStrings, Measures
using KernelDensity, Statistics
using Colors, ColorSchemes

# Resolve namespace conflicts by being explicit
import Plots: plot, plot!, scatter, scatter!, histogram, contour!, vline!, vspan!, annotate!
import Plots: xlabel!, ylabel!, title!, xticks!, xlims!, ylims!, xaxis, yaxis, framestyle
import Plots: default, savefig, layout, size, dpi, linewidth, alpha, color, markersize
import StatsPlots: fit, Histogram

# Set publication-quality plotting defaults
Plots.default(fontfamily="Computer Modern", 
        linewidth=2, 
        framestyle=:box, 
        label=nothing, 
        grid=false,
        tickfontsize=12,
        guidefontsize=14,
        titlefontsize=16,
        legendfontsize=12,
        left_margin=5mm,
        bottom_margin=5mm,
        top_margin=3mm,
        right_margin=3mm)

# Define colors for consistency
color_primary = RGB(0.2, 0.4, 0.8)    # Blue
color_secondary = RGB(0.8, 0.2, 0.2)  # Red
color_tertiary = RGB(0.2, 0.6, 0.2)   # Green
color_gray = RGB(0.5, 0.5, 0.5)       # Gray

# Function to create corner plot (pair plot)
function create_corner_plot(chain, derived_params)
    # Extract data
    M_wd = chain[:M_wd].data[:]
    M_giant = chain[:M_giant].data[:]
    i_deg = chain[:i].data[:]
    q_ratio = derived_params.q
    total_mass = M_wd .+ M_giant
    
    # Create parameter arrays and labels
    params = [M_wd, M_giant, i_deg, q_ratio, total_mass]
    param_labels = [L"M_{\mathrm{WD}}\,(\mathrm{M}_\odot)", 
                   L"M_{\mathrm{giant}}\,(\mathrm{M}_\odot)", 
                   L"i\,(\mathrm{deg})", 
                   L"q = M_{\mathrm{WD}}/M_{\mathrm{giant}}", 
                   L"M_{\mathrm{total}}\,(\mathrm{M}_\odot)"]
    
    n_params = length(params)
    
    # Create subplot layout
    plots_array = Array{Any}(undef, n_params, n_params)
    
    for i in 1:n_params
        for j in 1:n_params
            if i == j
                # Diagonal: 1D histograms
                p = Plots.histogram(params[i], 
                             bins=30, 
                             normalize=:pdf,
                             alpha=0.7,
                             color=color_primary,
                             linewidth=0,
                             label="")
                
                # Add KDE overlay
                kde_result = kde(params[i])
                Plots.plot!(p, kde_result.x, kde_result.density, 
                     linewidth=3, 
                     color=:black,
                     linestyle=:solid,
                     label="")
                
                # Format axes
                if i == n_params
                    Plots.xlabel!(p, param_labels[i])
                else
                    Plots.plot!(p, xaxis=false)
                end
                Plots.ylabel!(p, "Density")
                
            elseif i > j
                # Lower triangle: 2D scatter plots with contours
                p = Plots.scatter(params[j], params[i], 
                           alpha=0.3, 
                           markersize=1.5,
                           color=color_primary,
                           markerstrokewidth=0,
                           label="")
                
                # Add contour levels (simplified approach)
                try
                    # Create 2D histogram for contours
                    h = StatsPlots.fit(StatsPlots.Histogram, (params[j], params[i]), nbins=15)
                    x_edges = h.edges[1]
                    y_edges = h.edges[2]
                    z = h.weights
                    
                    # Calculate contour levels (68%, 95% confidence)
                    sorted_z = sort(vec(z), rev=true)
                    total_counts = sum(sorted_z)
                    cum_counts = cumsum(sorted_z)
                    
                    idx_68 = findfirst(cum_counts .>= 0.68 * total_counts)
                    idx_95 = findfirst(cum_counts .>= 0.95 * total_counts)
                    
                    if idx_68 !== nothing && idx_95 !== nothing
                        level_68 = sorted_z[idx_68]
                        level_95 = sorted_z[idx_95]
                        
                        # Create smooth contour
                        x_centers = (x_edges[1:end-1] .+ x_edges[2:end]) ./ 2
                        y_centers = (y_edges[1:end-1] .+ y_edges[2:end]) ./ 2
                        
                        Plots.contour!(p, x_centers, y_centers, z', 
                                levels=[level_95, level_68],
                                colors=[color_secondary, color_tertiary],
                                linewidth=2,
                                label="")
                    end
                catch
                    # If contour fails, continue without it
                end
                
                # Format axes
                if i == n_params
                    Plots.xlabel!(p, param_labels[j])
                else
                    Plots.plot!(p, xaxis=false)
                end
                if j == 1
                    Plots.ylabel!(p, param_labels[i])
                else
                    Plots.plot!(p, yaxis=false)
                end
                
            else
                # Upper triangle: correlation coefficients
                p = Plots.plot(xlims=(0,1), ylims=(0,1), 
                        xaxis=false, yaxis=false,
                        grid=false, 
                        framestyle=:none,
                        label="")
                
                # Add correlation coefficient
                if i != j
                    corr_val = cor(params[j], params[i])
                    Plots.annotate!(p, 0.5, 0.5, 
                             Plots.text(string("r = ", round(corr_val, digits=3)), 
                                  :center, 14))
                end
            end
            
            plots_array[i, j] = p
        end
    end
    
    # Combine all plots
    corner_plot = Plots.plot(plots_array..., 
                      layout=(n_params, n_params),
                      size=(1000, 1000),
                      dpi=300)
    
    return corner_plot
end

# Function to create mass function comparison plot
function create_mass_function_plot(derived_params, fm_obs, sigma_fm)
    p = Plots.histogram(derived_params.fm, 
                 bins=30,
                 normalize=:pdf,
                 alpha=0.7,
                 color=color_primary,
                 label="Predicted f(m)")
    
    # Add KDE
    kde_fm = kde(derived_params.fm)
    Plots.plot!(p, kde_fm.x, kde_fm.density,
          linewidth=3,
          color=:black,
          label="PDF")
    
    # Add observed value with error bar
    y_max = maximum(kde_fm.density)
    Plots.vline!(p, [fm_obs], 
           linewidth=3, 
           color=color_secondary,
           label="Observed f(m)")
    
    # Add error region
    Plots.vspan!(p, [fm_obs - sigma_fm, fm_obs + sigma_fm],
           alpha=0.3,
           color=color_secondary,
           label=L"1\sigma\,\mathrm{uncertainty}")
    
    Plots.xlabel!(p, L"f(m)\,(\mathrm{M}_\odot)")
    Plots.ylabel!(p, "Probability Density")
    Plots.title!(p, "Mass Function Comparison")
    
    # Add statistics text
    mean_fm = mean(derived_params.fm)
    std_fm = std(derived_params.fm)
    Plots.annotate!(p, Plots.xlims(p)[2] * 0.98, Plots.ylims(p)[2] * 0.95, 
             Plots.text(string("Predicted: ", round(mean_fm, digits=4), " ± ", round(std_fm, digits=4)), 
                  :right, 12))
    Plots.annotate!(p, Plots.xlims(p)[2] * 0.98, Plots.ylims(p)[2] * 0.85, 
             Plots.text(string("Observed: ", fm_obs, " ± ", sigma_fm), 
                  :right, 12))
    
    return p
end

# Function to create parameter summary plot
function create_parameter_summary(chain, derived_params)
    # Calculate statistics
    params_data = [
        chain[:M_wd].data[:],
        chain[:M_giant].data[:],
        chain[:i].data[:],
        derived_params.q,
        derived_params.a
    ]
    
    param_labels = [L"M_{\mathrm{WD}}\,(\mathrm{M}_\odot)", 
                   L"M_{\mathrm{giant}}\,(\mathrm{M}_\odot)", 
                   L"i\,(\mathrm{deg})", 
                   L"q", 
                   L"a\,(\mathrm{AU})"]
    
    means = [mean(p) for p in params_data]
    stds = [std(p) for p in params_data]
    q16 = [quantile(p, 0.16) for p in params_data]
    q84 = [quantile(p, 0.84) for p in params_data]
    
    # Create error bar plot
    p = Plots.scatter(1:5, means, 
               yerror=(means .- q16, q84 .- means),
               markersize=8,
               color=color_primary,
               markerstrokewidth=2,
               markerstrokecolor=:black,
               linewidth=3,
               label="")
    
    # Format plot
    Plots.xticks!(p, 1:5, param_labels)
    Plots.ylabel!(p, "Parameter Value")
    Plots.title!(p, "Parameter Estimates with 68% Confidence Intervals")
    
    # Rotate x labels for better readability
    Plots.plot!(p, xrotation=45)
    
    return p
end

# Function to create trace plots for MCMC diagnostics
function create_trace_plots(chain)
    n_samples = size(chain, 1)
    iterations = 1:n_samples
    
    # Create individual trace plots
    p1 = Plots.plot(iterations, chain[:M_wd].data[:],
              color=color_primary,
              alpha=0.8,
              ylabel=L"M_{\mathrm{WD}}\,(\mathrm{M}_\odot)",
              title="White Dwarf Mass",
              label="")
    
    p2 = Plots.plot(iterations, chain[:M_giant].data[:],
              color=color_secondary,
              alpha=0.8,
              ylabel=L"M_{\mathrm{giant}}\,(\mathrm{M}_\odot)",
              title="Giant Mass",
              label="")
    
    p3 = Plots.plot(iterations, chain[:i].data[:],
              color=color_tertiary,
              alpha=0.8,
              xlabel="Iteration",
              ylabel=L"i\,(\mathrm{deg})",
              title="Inclination",
              label="")
    
    # Combine plots
    trace_plot = Plots.plot(p1, p2, p3,
                     layout=(3, 1),
                     size=(800, 600),
                     dpi=300)
    
    return trace_plot
end

# Main plotting function
function create_all_publication_plots(chain, derived_params, fm_obs, sigma_fm)
    println("Creating publication-quality plots...")
    
    # 1. Corner plot
    println("  - Creating corner plot...")
    corner_p = create_corner_plot(chain, derived_params)
    
    # 2. Mass function comparison
    println("  - Creating mass function comparison...")
    mf_p = create_mass_function_plot(derived_params, fm_obs, sigma_fm)
    
    # 3. Parameter summary
    println("  - Creating parameter summary...")
    summary_p = create_parameter_summary(chain, derived_params)
    
    # 4. Trace plots
    println("  - Creating trace plots...")
    trace_p = create_trace_plots(chain)
    
    # Save plots
    println("  - Saving plots...")
    Plots.savefig(corner_p, "symbiotic_binary_corner_plot.pdf")
    Plots.savefig(corner_p, "symbiotic_binary_corner_plot.png")
    
    Plots.savefig(mf_p, "mass_function_comparison.pdf")
    Plots.savefig(mf_p, "mass_function_comparison.png")
    
    Plots.savefig(summary_p, "parameter_summary.pdf")
    Plots.savefig(summary_p, "parameter_summary.png")
    
    Plots.savefig(trace_p, "mcmc_traces.pdf")
    Plots.savefig(trace_p, "mcmc_traces.png")
    
    println("All plots saved successfully!")
    
    return (corner=corner_p, mass_function=mf_p, summary=summary_p, traces=trace_p)
end

# Create a combined figure for the paper
function create_combined_figure(chain, derived_params, fm_obs, sigma_fm)
    # Create individual plots with smaller sizes
    mf_p = create_mass_function_plot(derived_params, fm_obs, sigma_fm)
    summary_p = create_parameter_summary(chain, derived_params)
    
    # Combine into a single figure
    combined = Plots.plot(mf_p, summary_p,
                   layout=(1, 2),
                   size=(1200, 400),
                   dpi=300)
    
    Plots.savefig(combined, "symbiotic_binary_combined.pdf")
    Plots.savefig(combined, "symbiotic_binary_combined.png")
    
    return combined
end

println("Publication plotting functions loaded successfully!")
println("Usage:")
println("  plots = create_all_publication_plots(chain, derived, fm_obs, sigma_fm)")
println("  combined_fig = create_combined_figure(chain, derived, fm_obs, sigma_fm)")