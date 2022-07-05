# Proper Orthogonal Decomposition

The following fluid dynamics example is adapted from this 
[YouTube video](https://youtu.be/F7rWoxeGrko) which demonstrates the the Stable Fluids 
algorithm for a 2d fluid flow. 

```@example pod
using FFTW, Plots, Interpolations, LinearAlgebra
N_POINTS = 250
KINEMATIC_VISCOSITY = 0.0001
TIME_STEP_LENGTH = 0.01
N_TIME_STEPS = 100
ELEMENT_LENGTH = 1.0 / (N_POINTS - 1)
X_INTERVAL = 0.0:ELEMENT_LENGTH:1.0
Y_INTERVAL = 0.0:ELEMENT_LENGTH:1.0
function backtrace!(backtraced_positions, original_positions, direction)
    backtraced_positions[:] = mod1.(original_positions - TIME_STEP_LENGTH * direction, 1.0)
end
function interpolate_positions!(field_interpolated, field, interval_x, interval_y,
                                query_points_x, query_points_y)
    interpolator = LinearInterpolation((interval_x, interval_y), field)
    field_interpolated[:] = interpolator.(query_points_x, query_points_y)
end
function stable_fluids_fft()
    coordinates_x = [x for x in X_INTERVAL, y in Y_INTERVAL]
    coordinates_y = [y for x in X_INTERVAL, y in Y_INTERVAL]
    wavenumbers_1d = fftfreq(N_POINTS) .* N_POINTS
    wavenumbers_x = [k_x for k_x in wavenumbers_1d, k_y in wavenumbers_1d]
    wavenumbers_y = [k_y for k_x in wavenumbers_1d, k_y in wavenumbers_1d]
    wavenumbers_norm = [norm([k_x, k_y]) for k_x in wavenumbers_1d, k_y in wavenumbers_1d]
    decay = exp.(-TIME_STEP_LENGTH .* KINEMATIC_VISCOSITY .* wavenumbers_norm .^ 2)
    wavenumbers_norm[iszero.(wavenumbers_norm)] .= 1.0
    normalized_wavenumbers_x = wavenumbers_x ./ wavenumbers_norm
    normalized_wavenumbers_y = wavenumbers_y ./ wavenumbers_norm
    force_x = 100.0 .* (exp.(-1.0 / (2 * 0.005) *
                    ((coordinates_x .- 0.2) .^ 2 + (coordinates_y .- 0.45) .^ 2))
               -
               exp.(-1.0 / (2 * 0.005) *
                    ((coordinates_x .- 0.8) .^ 2 + (coordinates_y .- 0.55) .^ 2)))
    backtraced_coordinates_x = similar(coordinates_x)
    backtraced_coordinates_y = similar(coordinates_y)
    velocity_x = similar(coordinates_x)
    velocity_y = similar(coordinates_y)
    velocity_x_prev = similar(velocity_x)
    velocity_y_prev = similar(velocity_y)
    velocity_x_fft = similar(velocity_x)
    velocity_y_fft = similar(velocity_y)
    pressure_fft = similar(coordinates_x)
    curls = similar(velocity_x, length(X_INTERVAL) - 1, length(Y_INTERVAL) - 1,
                    N_TIME_STEPS)
    for iter in 1:N_TIME_STEPS
        time_current = (iter - 1) * TIME_STEP_LENGTH
        pre_factor = max(1 - time_current, 0.0)
        velocity_x_prev += TIME_STEP_LENGTH * pre_factor * force_x
        backtrace!(backtraced_coordinates_x, coordinates_x, velocity_x_prev)
        backtrace!(backtraced_coordinates_y, coordinates_y, velocity_y_prev)
        interpolate_positions!(velocity_x, velocity_x_prev, X_INTERVAL, Y_INTERVAL,
                               backtraced_coordinates_x, backtraced_coordinates_y)
        interpolate_positions!(velocity_y, velocity_y_prev, X_INTERVAL, Y_INTERVAL,
                               backtraced_coordinates_x, backtraced_coordinates_y)
        velocity_x_fft = fft(velocity_x)
        velocity_y_fft = fft(velocity_y)
        velocity_x_fft .*= decay
        velocity_y_fft .*= decay
        pressure_fft = (velocity_x_fft .* normalized_wavenumbers_x
                        +
                        velocity_y_fft .* normalized_wavenumbers_y)
        velocity_x_fft -= pressure_fft .* normalized_wavenumbers_x
        velocity_y_fft -= pressure_fft .* normalized_wavenumbers_y
        velocity_x = real(ifft(velocity_x_fft))
        velocity_y = real(ifft(velocity_y_fft))
        velocity_x_prev = velocity_x
        velocity_y_prev = velocity_y
        d_u__d_y = diff(velocity_x, dims = 2)[2:end, :]
        d_v__d_x = diff(velocity_y, dims = 1)[:, 2:end]
        curls[:, :, iter] = d_u__d_y - d_v__d_x
    end
    return curls
end
curls = stable_fluids_fft(); nothing
```

We got the data of curl at a few time steps. The time-averaged field is visualized as 
follows.

```@example pod
using Statistics
time_averaged = dropdims(mean(curls; dims = 3); dims = 3)
heatmap(X_INTERVAL, Y_INTERVAL, time_averaged', aspect_ratio = :equal, size = (1700, 1600),
        xlabel = "x", ylabel = "y")
```

We subtract the time-averaged field from each individual snapshot and change the shape to 
our snapshot matrix:

```@example pod
snapshots = curls .- time_averaged
snapshots = reshape(snapshots, :, N_TIME_STEPS)
```

```math
D =
\begin{pmatrix}
  d_{11} & d_{12} & \cdots & d_{1m} \\
  \vdots & \vdots & & \vdots \\
  d_{nm} & d_{n2} & \cdots & d_{nm}
\end{pmatrix}
=
\begin{pmatrix}
  d(x_1,y_1,t_1) & d(x_1,y_1,t_2) & \cdots & d(x_1,y_1,t_m) \\
  \vdots & \vdots & & \vdots \\
  d(x_{N_x},y_{N_y},t_1) & d(x_{N_x},y_{N_y},t_2) & \cdots & d(x_{N_x},y_{N_y},t_m)
\end{pmatrix}
```

where each column represents one snapshot. ``n`` is the number of discrete points and ``m``
is the number of snapshots. Each element ``d_{ij}`` of ``D`` is the scalar value at point 
``i`` measured at time ``j``. 

Then we compute the POD with dimension, say, 3.

```@example pod
using ModelOrderReduction
pod_dim = 3
pod_basis, singular_vals = pod(snapshots, pod_dim); nothing
```

As an illustration, let's see the first 3 POD modes.

```@example pod
mode_plots = Any[]
for dim in 1:pod_dim
    mode = reshape((@view pod_basis[:, dim]),
                   (length(X_INTERVAL) - 1, length(Y_INTERVAL) - 1))
    push!(mode_plots,
          heatmap(X_INTERVAL, Y_INTERVAL, mode', aspect_ratio = :equal,
                  title = "mode $dim", xlabel = "x", ylabel = "y"))
    push!(mode_plots,
          plot(vec((@view pod_basis[:, dim])' * snapshots), ylim = (-15, 15),
               title = "mode $dim", xlabel = "t", ylabel = "a", lw = 3))
end
plot(mode_plots..., layout = (pod_dim, 2), legend = false, size = (3000, 2000))
```

The figures on the right show the time coefficient of the first 3 POD modes.
