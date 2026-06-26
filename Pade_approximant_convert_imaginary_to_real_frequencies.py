##############################################################################
################## Analytic continuation using Pade approximant ##############
################# Diem TX Dang, Dai-Nam Le and Lilia Woods ###################
###################### University of South Florida, USA ######################
##############################################################################

import numpy as np
import glob
import os
from scipy.optimize import minimize
import matplotlib
matplotlib.use('Agg')          # non-interactive backend, safe on HPC
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ===========================================================================
# Parameters
# ===========================================================================

ETA        = 0.001             # Lorentzian broadening (eV)
EPSILON_0  = 8.854187817e-22  # vacuum permittivity (F/Ang)

# Real-frequency output grid (eV).
# The grid runs from W_REAL_MIN to W_REAL_MAX with step DW_REAL.
# These values are used as defaults; they can be overridden from the
# command line with --w-min, --w-max, --dw.
W_REAL_MIN = 0.0
W_REAL_MAX = 1.0
DW_REAL    = 0.01

# ===========================================================================
# Core Pade engine
# ===========================================================================

def pade_self_consistent(iw, f_iw, w_real, eta=ETA):
    """
    Analytic continuation f(i*omega) -> f(omega) via two-point Pade.
    Both f(0) and the tail coefficient A are determined self-consistently
    by leave-one-out cross-validation on the imaginary axis.

    Parameters
    ----------
    iw     : 1D array, positive imaginary frequencies (real values, eV)
    f_iw   : 1D array, f(i*iw), complex
    w_real : 1D array, target real frequencies (eV)  -- arbitrary dense grid
    eta    : float, Lorentzian broadening for real-axis evaluation

    Returns
    -------
    out : ndarray, shape (len(w_real), 3)
          col 0 : real frequencies
          col 1 : Re f(omega + i*eta)
          col 2 : Im f(omega + i*eta)
    """

    # -----------------------------------------------------------------------
    # Scale to O(1) for numerical stability.
    # -----------------------------------------------------------------------
    f_scale = np.max(np.abs(f_iw))
    if f_scale == 0:
        raise ValueError("f_iw is identically zero.")
    f_iw_s = f_iw / f_scale

    # -----------------------------------------------------------------------
    # Thiele continued-fraction coefficients
    # -----------------------------------------------------------------------
    def thiele(z_pts, f_pts):
        N = len(z_pts)
        phi = np.zeros((N, N), dtype=complex)
        phi[:, 0] = f_pts
        for j in range(1, N):
            denom = phi[j:, j-1] - phi[j-1, j-1]
            tiny  = np.abs(phi[j-1, j-1]) * 1e-14 + 1e-300
            denom = np.where(np.abs(denom) < tiny, tiny, denom)
            phi[j:, j] = (z_pts[j:] - z_pts[j-1]) / denom
        a = phi.diagonal().copy()
        bad = ~np.isfinite(a)
        if np.any(bad):
            a[bad] = 0.0
        return a

    # -----------------------------------------------------------------------
    # Evaluate continued fraction
    # -----------------------------------------------------------------------
    def eval_cf(z, z_pts, a):
        result = np.zeros(len(z), dtype=complex)
        for i, zv in enumerate(z):
            cf = a[-1]
            for k in range(len(a) - 2, 0, -1):
                d = cf if np.abs(cf) > 1e-300 else 1e-300
                cf = a[k] + (zv - z_pts[k]) / d
                if not np.isfinite(cf):
                    cf = 1e300
            d = cf if np.abs(cf) > 1e-300 else 1e-300
            result[i] = a[0] + (zv - z_pts[0]) / d
        return result

    # -----------------------------------------------------------------------
    # Fit h(z) = [f(z) - f0] / [z^2 * f(z)] with Thiele Pade,
    # then recover f(z) = f0 / (1 - z^2 * h(z)).
    #
    # h is bounded on the imaginary axis (-> -f0/A as z -> inf) so the
    # continued fraction is stable.
    # -----------------------------------------------------------------------
    def fit_and_eval(z_fit, f_fit_s, z_eval, f0_s, A_s):
        denom_h = z_fit**2 * f_fit_s
        safe    = np.abs(denom_h) > 1e-30 * (np.max(np.abs(denom_h)) + 1e-300)
        z_fit, f_fit_s, denom_h = z_fit[safe], f_fit_s[safe], denom_h[safe]
        if len(z_fit) < 4:
            return np.full(len(z_eval), np.nan, dtype=complex)

        h = (f_fit_s - f0_s) / denom_h

        a      = thiele(z_fit, h)
        h_eval = eval_cf(z_eval, z_fit, a)

        near_zero_f = np.abs(z_eval) < 1e-15
        denom_f     = 1.0 - z_eval**2 * h_eval
        denom_f     = np.where(np.abs(denom_f) < 1e-30, 1e-30, denom_f)
        return np.where(near_zero_f, f0_s + 0.0j, f0_s / denom_f)

    # -----------------------------------------------------------------------
    # Leave-one-out cross-validation residual
    # -----------------------------------------------------------------------
    def residual(params):
        f0_s, A_s = params
        if A_s <= 0:
            return 1e10
        try:
            z_pts = 1j * iw
            total = 0.0
            for i in range(len(iw)):
                mask   = np.arange(len(iw)) != i
                f_pred = fit_and_eval(z_pts[mask], f_iw_s[mask],
                                      z_pts[[i]], f0_s, A_s)
                val = f_pred[0]
                total += 1e6 if not np.isfinite(val) else np.abs(val - f_iw_s[i])**2
            return float(np.real(total))
        except Exception:
            return 1e10

    # -----------------------------------------------------------------------
    # Initial guesses
    # -----------------------------------------------------------------------
    f0_s_init = float(np.real(f_iw_s[0]))
    n_tail    = max(5, len(iw) // 5)
    A_s_init  = float(np.mean(-iw[-n_tail:]**2 * np.real(f_iw_s[-n_tail:])))
    if A_s_init <= 0:
        A_s_init = abs(A_s_init) + 1e-6

    # -----------------------------------------------------------------------
    # Nelder-Mead optimisation
    # -----------------------------------------------------------------------
    opt = minimize(residual, [f0_s_init, A_s_init], method='Nelder-Mead',
                   options={'xatol': 1e-8, 'fatol': 1e-13, 'maxiter': 10000})
    f0_s_opt, A_s_opt = opt.x

    # -----------------------------------------------------------------------
    # Final continuation on the dense real-frequency grid
    # -----------------------------------------------------------------------
    f_real_s = fit_and_eval(1j * iw, f_iw_s, w_real + 1j * eta,
                            f0_s_opt, A_s_opt)
    f_real   = f_real_s * f_scale

    return np.column_stack([w_real, f_real.real, f_real.imag])


# ===========================================================================
# Plotting
# ===========================================================================

def plot_continuation(iw, f_iw, result, stem, folder):
    """
    Plot Re f(omega) (blue), Im f(omega) (orange), and f(i*omega) (black),
    all divided by epsilon_0, and save as <folder>/figs/<stem>.jpg.

    Parameters
    ----------
    iw     : 1D array, imaginary frequencies (eV)
    f_iw   : 1D array, f(i*omega), complex, in physical units (F/Ang)
    result : ndarray shape (N, 3), columns [omega, Re f, Im f] in F/Ang
    stem   : str, base filename without extension
    folder : str, parent folder; figure is saved into <folder>/figs/
    """
    fig_dir = os.path.join(folder, 'figs')
    os.makedirs(fig_dir, exist_ok=True)
    outpath = os.path.join(fig_dir, stem + '.jpg')

    w      = result[:, 0]
    re_f   = result[:, 1] / EPSILON_0
    im_f   = result[:, 2] / EPSILON_0
    f_imag = np.real(f_iw) / EPSILON_0

    fig, ax = plt.subplots(figsize=(7, 4.5))

    ax.plot(w,  re_f,   color='tab:blue',   lw=1.8,
            label=r'$\mathrm{Re}\, f(\omega)$')
    ax.plot(w,  im_f,   color='tab:orange', lw=1.8,
            label=r'$\mathrm{Im}\, f(\omega)$')
    ax.plot(iw, f_imag, color='black',      lw=1.4, ls='--',
            label=r'$f(i\omega)$')

    ylabel = r'$\mathrm{' + stem.replace('_', r'\_') + r'}\ /\ \varepsilon_0$'
    ax.set_xlabel(r'$\hbar\omega$ (eV)', fontsize=13)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(stem, fontsize=11)
    ax.axhline(0, color='gray', lw=0.6, ls=':')
    ax.legend(fontsize=11, framealpha=0.85)
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.tick_params(which='both', direction='in', top=True, right=True)

    fig.tight_layout()
    fig.savefig(outpath, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print("  Figure  : " + outpath)


# ===========================================================================
# Batch processing
# ===========================================================================

def process_all(folder='.', eta=ETA, pattern='*.dat', suffix='_real',
                w_real_min=W_REAL_MIN, w_real_max=W_REAL_MAX, dw_real=DW_REAL):
    """
    Find every file matching pattern in folder (excluding files that already
    contain suffix in their name), run analytic continuation on a uniform
    dense real-frequency grid, and write the result as <stem><suffix>.dat.

    Output column format:
        col 0 : real frequency  (uniform grid from w_real_min to w_real_max)
        col 1 : Re f(omega + i*eta)
        col 2 : Im f(omega + i*eta)

    Parameters
    ----------
    folder     : str,   directory to search (default: current directory)
    eta        : float, Lorentzian broadening in eV
    pattern    : str,   glob pattern for input files
    suffix     : str,   appended to the stem of each output filename
    w_real_min : float, lower bound of real-frequency grid (eV)
    w_real_max : float, upper bound of real-frequency grid (eV)
    dw_real    : float, step size of real-frequency grid (eV)
    """
    # Build the dense output grid once -- shared across all files
    w_real = np.arange(w_real_min, w_real_max + 0.5 * dw_real, dw_real)

    print("Real-frequency grid : {:.4f} to {:.4f} eV,  step {:.4f} eV,  {} points".format(
        w_real[0], w_real[-1], dw_real, len(w_real)))
    print("Broadening eta      : {} eV".format(eta))
    print()

    files = sorted(glob.glob(os.path.join(folder, pattern)))
    files = [f for f in files if suffix not in os.path.basename(f)]

    if not files:
        print("No files matching '" + pattern + "' found in '" + folder + "'.")
        return

    print("Found " + str(len(files)) + " file(s) to process.\n")

    for fpath in files:
        stem    = os.path.splitext(os.path.basename(fpath))[0]
        outname = os.path.join(folder, stem + suffix + '.dat')

        print("-" * 60)
        print("Input  : " + os.path.basename(fpath))
        print("Output : " + os.path.basename(outname))

        # Load
        try:
            data = np.loadtxt(fpath)
        except Exception as e:
            print("  ERROR loading file: " + str(e) + "  - skipped.\n")
            continue

        if data.ndim != 2 or data.shape[1] < 2:
            print("  ERROR: expected at least 2 columns - skipped.\n")
            continue

        iw   = data[:, 0]
        f_iw = data[:, 1].astype(complex)
        if data.shape[1] >= 3:
            f_iw += 1j * data[:, 2]

        print("  Points : " + str(len(iw)) +
              ",  iw in [" + str(round(iw[0], 3)) +
              ", " + str(round(iw[-1], 3)) + "] eV" +
              ",  |f| max = " + "{:.3e}".format(np.max(np.abs(f_iw))))

        # Run continuation (Pade works on f/EPSILON_0, which is O(1))
        try:
            result = pade_self_consistent(iw, f_iw / EPSILON_0, w_real, eta=eta)
            # Restore physical units
            result[:, 1] *= EPSILON_0
            result[:, 2] *= EPSILON_0
        except Exception as e:
            print("  ERROR during continuation: " + str(e) + "  - skipped.\n")
            continue

        n_bad = np.sum(~np.isfinite(result[:, 1:]))
        if n_bad:
            print("  Warning: " + str(n_bad) + " non-finite values in output.")

        # Read any header lines from the input file
        header_lines = []
        try:
            with open(fpath, 'r') as fh:
                for line in fh:
                    if line.strip().startswith('#'):
                        header_lines.append(line.rstrip())
                    else:
                        break
        except Exception:
            pass

        # Save output
        with open(outname, 'w') as fh:
            for hl in header_lines:
                fh.write(hl + '\n')
            for row in result:
                fh.write("  {:18.8E}  {:18.8E}  {:18.8E}\n".format(
                    row[0], row[1], row[2]))

        print("  Written " + str(len(result)) + " lines -> " + outname)

        # Plot
        plot_continuation(iw, f_iw, result, stem, folder)
        print()

    print("Done.")


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Analytic continuation of polarizability from i*omega to omega.')
    parser.add_argument('folder', nargs='?', default='.',
                        help='Folder containing input .dat files (default: .)')
    parser.add_argument('--eta', type=float, default=ETA,
                        help='Broadening eta in eV (default: ' + str(ETA) + ')')
    parser.add_argument('--pattern', default='*.dat',
                        help='Glob pattern for input files (default: *.dat)')
    parser.add_argument('--suffix', default='_real',
                        help='Suffix appended to output filenames (default: _real)')
    parser.add_argument('--w-min', dest='w_real_min', type=float, default=W_REAL_MIN,
                        help='Lower bound of real-frequency grid in eV (default: '
                             + str(W_REAL_MIN) + ')')
    parser.add_argument('--w-max', dest='w_real_max', type=float, default=W_REAL_MAX,
                        help='Upper bound of real-frequency grid in eV (default: '
                             + str(W_REAL_MAX) + ')')
    parser.add_argument('--dw', dest='dw_real', type=float, default=DW_REAL,
                        help='Step size of real-frequency grid in eV (default: '
                             + str(DW_REAL) + ')')
    args = parser.parse_args()

    process_all(folder=args.folder, eta=args.eta,
                pattern=args.pattern, suffix=args.suffix,
                w_real_min=args.w_real_min, w_real_max=args.w_real_max,
                dw_real=args.dw_real)