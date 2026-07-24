"""
%%writefile particle_filter_auv.py
!python particle_filter_auv.py --show-all

or
import particle_filter_auv
particle_filter_auv.main(['--show-all'])

or

import particle_filter_auv
particle_filter_auv.main(['--animate'])


particle_filter_auv.main(['--show-likelihood', '--step', '20'])

particle_filter_auv.main(['--animate', '--show-likelihood', '--show-neff'])

or
particle_filter_auv.main(['--export-figures', './figures', '--seed', '7'])


Particle Filter Tutorial — Tracking a Drifting AUV with a Noisy Acoustic Sensor
=================================================================================

This is a refined Python port of the classic Gordon/Salmond/Smith (1993)
particle-filter benchmark, restructured around a physically motivated
scenario (see README.md) instead of the original hand-wavy story.

Run modes
---------
    python particle_filter_auv.py                     # basic run + summary plot
    python particle_filter_auv.py --animate            # movie-like live plot
    python particle_filter_auv.py --show-init-dist     # initial prior particle cloud
    python particle_filter_auv.py --show-weight-space   --step 20
    python particle_filter_auv.py --show-stage-plots    --step 20
    python particle_filter_auv.py --show-all
    python particle_filter_auv.py --export-figures ./figures   # headless PNG export for docs

All of the plots that were commented out (%{ ... %}) in the original MATLAB
script are reproduced here behind explicit flags instead of comments, so you
can turn them on/off without editing the code.
"""

import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


# ----------------------------------------------------------------------------
# 1. Model definition  (state equation f, measurement equation h)
# ----------------------------------------------------------------------------
def f_state(x, t, noise_std=0.0, rng=None):
    """State transition: x_t = f(x_{t-1}, t) + w_t.

    The nonlinear/oscillatory form stands in for a drifting vehicle's true
    dynamics (damping + restoring nonlinearity + periodic current forcing).
    Works for scalars or numpy arrays (vectorized over particles).
    """
    x = np.asarray(x, dtype=float)
    drift = 0.5 * x + 25.0 * x / (1.0 + x ** 2) + 8.0 * np.cos(1.2 * (t - 1))
    if noise_std > 0:
        gen = rng if rng is not None else np.random.default_rng()
        w = gen.standard_normal(x.shape) * noise_std
        return drift + w
    return drift


def h_meas(x):
    """Measurement model: z_t = h(x_t) + v_t.  h(x) = x^2 / 20."""
    x = np.asarray(x, dtype=float)
    return x ** 2 / 20.0


def gaussian_likelihood(z, z_hat, R):
    """p(z | x) assuming Gaussian measurement noise N(0, R)."""
    return (1.0 / np.sqrt(2 * np.pi * R)) * np.exp(-((z - z_hat) ** 2) / (2 * R))


# ----------------------------------------------------------------------------
# 2. Resampling schemes
# ----------------------------------------------------------------------------
def multinomial_resample(weights, rng):
    N = len(weights)
    cdf = np.cumsum(weights)
    cdf[-1] = 1.0
    u = rng.random(N)
    return np.searchsorted(cdf, u)


def systematic_resample(weights, rng):
    """Lower-variance resampling than multinomial; a single random offset
    with N evenly spaced draws. This is the standard fix used in modern
    particle-filter implementations to reduce resampling noise."""
    N = len(weights)
    positions = (np.arange(N) + rng.random()) / N
    cdf = np.cumsum(weights)
    cdf[-1] = 1.0
    return np.searchsorted(cdf, positions)


def effective_sample_size(weights):
    """N_eff ~ how many particles are 'meaningfully' contributing.
    N_eff = 1 / sum(w_i^2). Drops toward 1 when weights collapse onto a
    single particle (degeneracy)."""
    return 1.0 / np.sum(weights ** 2)


# ----------------------------------------------------------------------------
# 3. Particle filter core loop
# ----------------------------------------------------------------------------
class ParticleFilterRun:
    """Runs the filter once and stores everything needed for every plot
    variant (raw / weighted / resampled, weight-space, animation, etc.)."""

    def __init__(self, N=200, T=75, x0=0.1, x_N=1.0, x_R=1.0, V=2.0,
                 resample_method="systematic", estimate_method="weighted_mean",
                 resample_threshold=0.5, roughening=True, seed=None):
        self.N, self.T = N, T
        self.x_N, self.x_R, self.V = x_N, x_R, V
        self.resample_method = resample_method
        self.estimate_method = estimate_method
        self.resample_threshold = resample_threshold  # fraction of N; resample if N_eff below this*N
        self.roughening = roughening
        self.rng = np.random.default_rng(seed)

        # --- true system & initial prior particle cloud ---
        self.x_true = x0
        self.x_P = x0 + np.sqrt(V) * self.rng.standard_normal(N)   # prior particles at t=0
        self.init_particles = self.x_P.copy()                       # kept for the "initial dist" plot

        # storage across time, for every requested plot
        self.x_out = [x0]
        self.z_out = [h_meas(x0) + np.sqrt(x_R) * self.rng.standard_normal()]
        self.x_est_out = [x0]
        self.neff_out = [N]
        self.resampled_flags = [False]

        # per-step snapshots (only kept if a caller asks for a specific step)
        self.snapshots = {}

    def step(self, t):
        rng = self.rng
        # ---- (a) advance the TRUE hidden state and take a real measurement ----
        self.x_true = f_state(self.x_true, t, np.sqrt(self.x_N), rng)
        z = h_meas(self.x_true) + np.sqrt(self.x_R) * rng.standard_normal()

        # ---- (b) PREDICT: push every particle through the state equation ----
        x_P_update = f_state(self.x_P, t, np.sqrt(self.x_N), rng)   # "raw estimates"

        # ---- (c) predicted measurement for each particle ----
        z_update = h_meas(x_P_update)

        # ---- (d) WEIGHT: likelihood of the real measurement under each particle ----
        w = gaussian_likelihood(z, z_update, self.x_R)
        w_sum = np.sum(w)
        if w_sum <= 0 or not np.isfinite(w_sum):
            # numerical fallback: all particles equally implausible -> equal weights
            w = np.ones(self.N) / self.N
        else:
            w = w / w_sum                                            # normalize -> "weighted estimates"

        # ---- (e) point estimate BEFORE resampling (weighted mean is the
        #          proper Monte-Carlo estimator of E[x_t | z_1:t]) ----
        if self.estimate_method == "weighted_mean":
            x_est = np.sum(w * x_P_update)
        else:
            x_est = np.mean(x_P_update)

        # ---- (f) RESAMPLE only when weights have degenerated ----
        n_eff = effective_sample_size(w)
        do_resample = n_eff < self.resample_threshold * self.N
        if do_resample:
            if self.resample_method == "systematic":
                idx = systematic_resample(w, self.rng)
            else:
                idx = multinomial_resample(w, self.rng)
            x_P_new = x_P_update[idx]
            if self.roughening:
                # small jitter proportional to the particle spread, prevents
                # sample impoverishment (many identical copies after resampling)
                sigma = self.V ** 0.5 * self.N ** (-1.0 / 1.0) * 0.2 * (np.std(x_P_update) + 1e-6)
                x_P_new = x_P_new + sigma * self.rng.standard_normal(self.N)
        else:
            x_P_new = x_P_update  # keep weighted particles as-is, no resampling noise added

        # ---- snapshot bookkeeping ----
        self.snapshots[t] = dict(
            x_P_update=x_P_update.copy(), z_update=z_update.copy(),
            w=w.copy(), z=z, x_true=self.x_true, x_P_resampled=x_P_new.copy(),
            resampled=do_resample, x_est=x_est, n_eff=n_eff,
        )

        self.x_P = x_P_new
        self.x_out.append(self.x_true)
        self.z_out.append(z)
        self.x_est_out.append(x_est)
        self.neff_out.append(n_eff)
        self.resampled_flags.append(do_resample)
        return self.snapshots[t]

    def run(self):
        for t in range(1, self.T + 1):
            self.step(t)
        return self


# ----------------------------------------------------------------------------
# 4. Plot builders  (each corresponds to a commented block in the MATLAB file)
# ----------------------------------------------------------------------------
def plot_initial_distribution(pf, save=None):
    """Reproduces the commented-out MATLAB block:
    'show the distribution the particles around this initial value of x.'"""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(np.ones_like(pf.init_particles), pf.init_particles, '.k', markersize=6)
    axes[0].set_xlabel("time step"); axes[0].set_ylabel("flight/vehicle position")
    axes[0].set_title("Prior particles at t=0")
    axes[1].hist(pf.init_particles, bins=30, color="0.4")
    axes[1].set_xlabel("position"); axes[1].set_ylabel("count")
    axes[1].set_title(f"Histogram (N={pf.N}) — approximates N(x0, V={pf.V})")
    fig.suptitle("Initial prior: particles sampled from a Gaussian around x0")
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=130)
    return fig


def plot_weight_space(pf, t, save=None):
    """Reproduces the commented block that plots weight magnitude against
    z_update and against x_P_update — i.e. shows HOW weights get assigned."""
    snap = pf.snapshots[t]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    axes[0].plot(snap["w"], snap["z_update"], '.k', markersize=6)
    axes[0].plot(0, snap["z"], '.r', markersize=20, label="actual measurement z")
    axes[0].set_xlabel("weight magnitude"); axes[0].set_ylabel("predicted measurement (z_update)")
    axes[0].set_title(f"t={t}: weight vs. predicted measurement")
    axes[0].legend()

    axes[1].plot(snap["w"], snap["x_P_update"], '.k', markersize=6)
    axes[1].plot(0, snap["x_true"], '.r', markersize=20, label="true state x")
    axes[1].set_xlabel("weight magnitude"); axes[1].set_ylabel("predicted particle position")
    axes[1].set_title(f"t={t}: weight vs. predicted state")
    axes[1].legend()

    fig.suptitle("Weight magnitude = Gaussian likelihood of z given each particle's predicted measurement")
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=130)
    return fig


def plot_stage_comparison(pf, t, save=None):
    """Raw estimates -> weighted estimates -> weighted resampling, the
    three-panel evolution from the commented MATLAB block."""
    snap = pf.snapshots[t]
    x_raw = snap["x_P_update"]
    w = snap["w"]
    x_res = snap["x_P_resampled"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=True)

    axes[0].plot(np.zeros_like(x_raw), x_raw, '.k', markersize=6)
    axes[0].set_title("1. Raw estimates\n(prediction only, no data yet)")
    axes[0].set_xlabel("fixed x-axis (visual spread only)")
    axes[0].set_ylabel("particle position")

    sizes = 20 + 2000 * w  # scale marker size by weight so high-weight ones pop
    sc = axes[1].scatter(np.zeros_like(x_raw), x_raw, s=sizes, c=w, cmap="viridis", edgecolor="k", linewidth=0.3)
    axes[1].plot(0, snap["x_true"], '.r', markersize=22, label="true x")
    axes[1].set_title("2. Weighted estimates\n('filtered' by the measurement)")
    axes[1].legend()

    axes[2].plot(np.zeros_like(x_res), x_res, '.b', markersize=5, alpha=0.6, label="resampled particles")
    axes[2].plot(0, snap["x_est"], '.g', markersize=24, label="point estimate (mean)")
    axes[2].plot(0, snap["x_true"], '.r', markersize=18, label="true x")
    axes[2].set_title("3. Weighted-based resampling\n(concentrated cloud)")
    axes[2].legend()

    fig.suptitle(f"t={t}: how the particle cloud is refined by one measurement "
                 f"(resampled={snap['resampled']}, N_eff={snap['n_eff']:.1f})")
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=130)
    return fig


def plot_likelihood_gaussian(pf, t, save=None):
    """Explicit picture of the Gaussian likelihood function itself: p(z|x)
    as a curve, with each particle's (z_update, weight) plotted on top."""
    snap = pf.snapshots[t]
    z_grid = np.linspace(snap["z"] - 4 * np.sqrt(pf.x_R), snap["z"] + 4 * np.sqrt(pf.x_R), 300)
    pdf = gaussian_likelihood(snap["z"], z_grid, pf.x_R)
    pdf_particles = gaussian_likelihood(snap["z"], snap["z_update"], pf.x_R)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(z_grid, pdf, '-', color="C0", label=r"$p(z_t \mid \hat z^{(i)}) $  (Gaussian, var=R)")
    ax.scatter(snap["z_update"], pdf_particles, color="k", s=15, zorder=5, label="each particle's predicted z")
    ax.axvline(snap["z"], color="r", linestyle="--", label="actual measurement z")
    ax.set_xlabel("measurement value z"); ax.set_ylabel("likelihood density")
    ax.set_title(f"t={t}: unnormalized weights ARE this Gaussian evaluated at each particle")
    ax.legend()
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=130)
    return fig


def plot_summary(pf, save=None):
    t_axis = np.arange(pf.T + 1)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(t_axis, pf.x_out, '.-b', label="true position", linewidth=1.5)
    ax.plot(t_axis, pf.x_est_out, '-.r', linewidth=2.5, label="particle filter estimate")
    ax.set_xlabel("time step"); ax.set_ylabel("vehicle position")
    rmse = np.sqrt(np.mean((np.array(pf.x_out) - np.array(pf.x_est_out)) ** 2))
    ax.set_title(f"AUV tracking via particle filter  (N={pf.N}, RMSE={rmse:.3f})")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=130)
    return fig, rmse


def plot_neff(pf, save=None):
    fig, ax = plt.subplots(figsize=(9, 3.5))
    ax.plot(pf.neff_out, '-k')
    ax.axhline(pf.resample_threshold * pf.N, color="r", linestyle="--",
               label=f"resample threshold ({pf.resample_threshold:.1f} N)")
    resample_steps = [i for i, r in enumerate(pf.resampled_flags) if r]
    ax.scatter(resample_steps, [pf.neff_out[i] for i in resample_steps],
               color="orange", zorder=5, s=15, label="resampling triggered")
    ax.set_xlabel("time step"); ax.set_ylabel("N_eff (effective sample size)")
    ax.set_title("Weight degeneracy: N_eff drops until a resample refreshes the cloud")
    ax.legend()
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=130)
    return fig


# ----------------------------------------------------------------------------
# 5. Movie-like live animation
# ----------------------------------------------------------------------------
def run_animation(pf_kwargs):
    """Runs the filter step-by-step and redraws the particle cloud + true
    trajectory live, like a movie, instead of only showing the final plot."""
    pf = ParticleFilterRun(**pf_kwargs)

    fig, (ax_traj, ax_cloud) = plt.subplots(1, 2, figsize=(12, 5))
    ax_traj.set_xlim(0, pf.T); 
    true_line, = ax_traj.plot([], [], '.-b', label="true position")
    est_line, = ax_traj.plot([], [], '-.r', linewidth=2, label="PF estimate")
    ax_traj.legend(); ax_traj.set_xlabel("time step"); ax_traj.set_ylabel("position")
    ax_traj.set_title("Trajectory (live)")

    cloud_scatter = ax_cloud.scatter([], [], s=10, c="k", alpha=0.5, label="particles")
    true_pt, = ax_cloud.plot([], [], '.r', markersize=16, label="true x")
    ax_cloud.set_xlim(-1, 1); ax_cloud.set_ylabel("position"); ax_cloud.set_xticks([])
    ax_cloud.set_title("Particle cloud (live)")
    ax_cloud.legend()

    def update(frame):
        if frame == 0:
            snap = dict(x_P_resampled=pf.init_particles, x_true=pf.x_out[0])
        else:
            snap = pf.step(frame)
        true_line.set_data(range(len(pf.x_out)), pf.x_out)
        est_line.set_data(range(len(pf.x_est_out)), pf.x_est_out)
        ax_traj.relim(); ax_traj.autoscale_view()

        cloud = snap.get("x_P_resampled", pf.x_P)
        cloud_scatter.set_offsets(np.column_stack([np.zeros_like(cloud), cloud]))
        true_pt.set_data([0], [snap["x_true"]])
        ax_cloud.relim(); ax_cloud.autoscale_view()
        fig.suptitle(f"t = {frame}")
        return true_line, est_line, cloud_scatter, true_pt

    ani = animation.FuncAnimation(fig, update, frames=range(0, pf.T + 1),
                                   interval=150, blit=False, repeat=False)

    # plt.show() only animates in a native desktop backend. Colab/Jupyter's
    # inline backend just renders a single static frame instead of playing
    # the movie, so in a notebook we instead render the animation as
    # embedded JS/HTML5, which Colab can actually play.
    in_notebook = False
    try:
        from IPython import get_ipython
        in_notebook = get_ipython() is not None
    except ImportError:
        pass

    if in_notebook:
        from IPython.display import HTML, display
        plt.close(fig)  # prevent the static duplicate frame from also rendering
        display(HTML(ani.to_jshtml()))
    else:
        plt.show()

    return ani


# ----------------------------------------------------------------------------
# 6. CLI
# ----------------------------------------------------------------------------
def main(argv=None):
    """argv=None reads from sys.argv (normal CLI use). Pass an explicit list
    (e.g. main(['--N', '500', '--animate'])) when calling from a notebook,
    since Jupyter/Colab injects its own '-f kernel.json' argument that would
    otherwise confuse argparse."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--N", type=int, default=200, help="number of particles")
    p.add_argument("--T", type=int, default=75, help="number of time steps")
    p.add_argument("--x0", type=float, default=0.1)
    p.add_argument("--xN", type=float, default=1.0, help="process noise variance")
    p.add_argument("--xR", type=float, default=1.0, help="measurement noise variance")
    p.add_argument("--V", type=float, default=2.0, help="initial prior variance")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--resample", choices=["systematic", "multinomial"], default="systematic")
    p.add_argument("--estimate", choices=["weighted_mean", "mean"], default="weighted_mean")
    p.add_argument("--resample-threshold", type=float, default=0.5,
                    help="resample when N_eff < threshold * N (adaptive resampling)")
    p.add_argument("--no-roughening", action="store_true", help="disable post-resample jitter")
    p.add_argument("--step", type=int, default=20, help="time step used for the per-step plots")

    p.add_argument("--animate", action="store_true", help="movie-like running animation")
    p.add_argument("--show-init-dist", action="store_true")
    p.add_argument("--show-weight-space", action="store_true")
    p.add_argument("--show-stage-plots", action="store_true")
    p.add_argument("--show-likelihood", action="store_true")
    p.add_argument("--show-neff", action="store_true")
    p.add_argument("--show-all", action="store_true")
    p.add_argument("--export-figures", type=str, default=None,
                    help="headless: save all figures as PNGs into this directory")

    if argv is None:
        argv = sys.argv[1:]
        # Jupyter/Colab auto-injects '-f <kernel.json>' when a script is run
        # as a cell (e.g. via %run or exec) — argparse doesn't know this flag
        # and errors out. Strip it automatically so the script "just works"
        # in a notebook without requiring an explicit argv list.
        if "-f" in argv:
            idx = argv.index("-f")
            argv = argv[:idx] + argv[idx + 2:]

    args = p.parse_args(argv)

    pf_kwargs = dict(N=args.N, T=args.T, x0=args.x0, x_N=args.xN, x_R=args.xR, V=args.V,
                      resample_method=args.resample, estimate_method=args.estimate,
                      resample_threshold=args.resample_threshold,
                      roughening=not args.no_roughening, seed=args.seed)

    if args.animate:
        run_animation(pf_kwargs)
        return

    pf = ParticleFilterRun(**pf_kwargs).run()
    fig_summary, rmse = plot_summary(pf)
    print(f"RMSE over {args.T} steps with N={args.N} particles: {rmse:.4f}")

    want_init = args.show_init_dist or args.show_all
    want_weight = args.show_weight_space or args.show_all
    want_stage = args.show_stage_plots or args.show_all
    want_like = args.show_likelihood or args.show_all
    want_neff = args.show_neff or args.show_all

    if args.export_figures:
        os.makedirs(args.export_figures, exist_ok=True)
        plt.close(fig_summary)
        plot_summary(pf, save=os.path.join(args.export_figures, "01_summary_trajectory.png"))
        plot_initial_distribution(pf, save=os.path.join(args.export_figures, "02_initial_distribution.png"))
        plot_stage_comparison(pf, args.step, save=os.path.join(args.export_figures, "03_stage_comparison.png"))
        plot_weight_space(pf, args.step, save=os.path.join(args.export_figures, "04_weight_space.png"))
        plot_likelihood_gaussian(pf, args.step, save=os.path.join(args.export_figures, "05_likelihood_gaussian.png"))
        plot_neff(pf, save=os.path.join(args.export_figures, "06_neff_degeneracy.png"))
        print(f"Figures written to {args.export_figures}/")
        return

    if want_init:
        plot_initial_distribution(pf)
    if want_stage:
        plot_stage_comparison(pf, args.step)
    if want_weight:
        plot_weight_space(pf, args.step)
    if want_like:
        plot_likelihood_gaussian(pf, args.step)
    if want_neff:
        plot_neff(pf)

    plt.show()


if __name__ == "__main__":
    main()
