##############################################################################
########### Rotatory power from dipole-quadrupole polarizability #############
################# Diem TX Dang, Dai-Nam Le and Lilia Woods ###################
###################### University of South Florida, USA ######################
##############################################################################

"""
Compute rotatory power components rho(omega) and theta(omega) from:

    rho(omega) + i*theta(omega) = (omega^2 / 6*c^2*epsilon_0) * [A^xyz(omega) - A^yxz(omega)]

where:
    - rho(omega)  : real part  [deg/mm]
    - theta(omega): imaginary part [deg/mm]
    - A^xyz, A^yxz: mixed dipole-quadrupole polarizability tensors [Farad]
    - omega       : angular frequency [rad/s], derived from hbar*omega [eV]
    - c           : speed of light in vacuum [m/s]
    - epsilon_0   : vacuum permittivity [F/m]

The raw result of the formula is in rad/mm (= mm^-1); this script converts
to deg/mm by multiplying by (180/pi).

Input files are auto-detected in the working directory by matching:
    {seedname}-polarizability-dq_xyz_real.dat
    {seedname}-polarizability-dq_yxz_real.dat

    Column 1 : hbar*omega  [eV]
    Column 2 : Re(A)       [F]
    Column 3 : Im(A)       [F]   (optional; assumed 0 if absent)

Outputs (one pair per seedname):
    {seedname}-rho_theta.dat   – data file  (cols: hbar*omega, rho, theta)
    {seedname}-rho_theta.png   – plot of rho (red) and theta (blue) vs hbar*omega

Usage
-----
    python compute_rho_theta.py                     # auto-detect in current dir
    python compute_rho_theta.py /path/to/dir        # auto-detect in given dir
    python compute_rho_theta.py xyz.dat yxz.dat     # explicit file paths
"""

# ===========================================================================
# Packages
# ===========================================================================

import numpy as np
import sys
import os
import glob
import matplotlib
matplotlib.use("Agg")                 # non-interactive backend (safe everywhere)
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ===========================================================================
# Physical constant - SI unit
# ===========================================================================
HBAR_SI         = 1.054571817e-34    # J·s
EV_TO_J         = 1.602176634e-19    # J/eV
C_SI            = 2.99792458e8       # m/s
EPS0_SI         = 8.8541878128e-12   # F/m

RAD_MM_TO_DEG_MM = 180.0 / np.pi    # rad/mm → deg/mm
M_INV_TO_MM_INV  = 1.0e-3           # m^-1   → mm^-1  (= rad/mm)

SUFFIX_XYZ = "-polarizability-dq_xyz_real.dat"
SUFFIX_YXZ = "-polarizability-dq_yxz_real.dat"


# ===========================================================================
# File detetion
# ===========================================================================

def find_file_pairs(search_dir="."):
    """
    Search *search_dir* for files matching
        {seedname}-polarizability-dq_xyz_real.dat
    and confirm a matching yxz partner exists.

    Returns a list of (seedname, xyz_path, yxz_path) tuples.
    """
    pattern   = os.path.join(search_dir, f"*{SUFFIX_XYZ}")
    xyz_files = sorted(glob.glob(pattern))

    pairs = []
    for xyz_path in xyz_files:
        basename = os.path.basename(xyz_path)
        seedname = basename[: -len(SUFFIX_XYZ)]
        yxz_path = os.path.join(search_dir, seedname + SUFFIX_YXZ)
        if os.path.isfile(yxz_path):
            pairs.append((seedname, xyz_path, yxz_path))
        else:
            print(f"  [warn] Found {basename} but no matching yxz file "
                  f"({os.path.basename(yxz_path)}) – skipping.")
    return pairs


def resolve_inputs(argv):
    """
    Parse command-line arguments and return a list of
    (seedname, xyz_path, yxz_path) tuples to process.

    Rules
    -----
    - No args          → auto-detect in current directory
    - 1 arg (a dir)    → auto-detect in that directory
    - 2 args (files)   → use those explicit paths
    """
    if len(argv) == 1:
        return find_file_pairs(".")

    if len(argv) == 2:
        arg = argv[1]
        if os.path.isdir(arg):
            return find_file_pairs(arg)
        else:
            print(f"Error: '{arg}' is not a directory.")
            sys.exit(1)

    if len(argv) == 3:
        xyz_path, yxz_path = argv[1], argv[2]
        for p in (xyz_path, yxz_path):
            if not os.path.isfile(p):
                print(f"Error: file not found: {p}")
                sys.exit(1)
        basename = os.path.basename(xyz_path)
        seedname = (basename[: -len(SUFFIX_XYZ)]
                    if basename.endswith(SUFFIX_XYZ)
                    else os.path.splitext(basename)[0])
        return [(seedname, xyz_path, yxz_path)]

    print(__doc__)
    sys.exit(1)


# ===========================================================================
# I/O subroutine
# ===========================================================================

def load_polarizability(filepath):
    """
    Load a polarizability file.

    Returns
    -------
    hw_eV : 1-D array          – hbar*omega [eV]
    A     : 1-D complex array  – polarizability [F]
              col2 + i*col3; col3 assumed zero when absent
    """
    data = np.loadtxt(filepath, comments="#")
    if data.ndim == 1:
        data = data[np.newaxis, :]

    hw_eV = data[:, 0]

    if data.shape[1] >= 3:
        A = data[:, 1] + 1j * data[:, 2]
    elif data.shape[1] == 2:
        print(f"  [info] Only 2 columns in {os.path.basename(filepath)}; "
              "treating A as purely real.")
        A = data[:, 1].astype(complex)
    else:
        raise ValueError(f"Unexpected column count ({data.shape[1]}) "
                         f"in {filepath}")
    return hw_eV, A


# ===========================================================================
# Plot subroutine
# ===========================================================================

def make_plot(seedname, hw_eV, rho, theta, out_dir):
    """
    Plot rho (red) and theta (blue) versus hbar*omega and save as PNG.
    """
    fig, ax = plt.subplots(figsize=(7, 4.5))

    ax.plot(hw_eV, rho,   color="red",  linewidth=1.5,
            label=r"$\rho(\omega)$")
    ax.plot(hw_eV, theta, color="blue", linewidth=1.5,
            label=r"$\theta(\omega)$")

    ax.axhline(0, color="black", linewidth=0.6, linestyle="--", alpha=0.5)

    ax.set_xlabel(r"$\hbar\omega$ (eV)", fontsize=13)
    ax.set_ylabel(r"$\rho,\;\theta$ (deg/mm)", fontsize=13)
    ax.set_title(seedname, fontsize=11, pad=8)

    ax.legend(fontsize=11, framealpha=0.85)
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.tick_params(which="both", direction="in",
                   top=True, right=True, labelsize=10)

    fig.tight_layout()

    plot_file = os.path.join(out_dir, f"{seedname}-rho_theta.png")
    fig.savefig(plot_file, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot     : {plot_file}")


# ===========================================================================
# Core calculation subroutine
# ===========================================================================

def compute_rho_theta(seedname, file_xyz, file_yxz):
    """
    Compute rho and theta for one seedname, write the data file, and plot.
    """
    print(f"\n{'─'*60}")
    print(f"  Seedname : {seedname}")
    print(f"  xyz file : {file_xyz}")
    print(f"  yxz file : {file_yxz}")

    hw_eV_xyz, A_xyz = load_polarizability(file_xyz)
    hw_eV_yxz, A_yxz = load_polarizability(file_yxz)

    if not np.allclose(hw_eV_xyz, hw_eV_yxz, rtol=1e-6):
        raise ValueError(
            f"hbar*omega grids for '{seedname}' do not match. "
            "Please verify both files share the same energy axis."
        )
    hw_eV = hw_eV_xyz

    # omega [rad/s]
    omega = hw_eV * EV_TO_J / HBAR_SI

    # prefactor  omega^2 / (6 c^2 eps0)  [m^-1 F^-1]
    prefactor = omega**2 / (6.0 * C_SI**2 * EPS0_SI)

    # rho + i*theta  [m^-1 = rad/m]
    result = prefactor * (A_xyz - A_yxz)

    # convert  rad/m → rad/mm → deg/mm
    conv    = M_INV_TO_MM_INV * RAD_MM_TO_DEG_MM
    rho     = np.real(result) * conv   # deg/mm
    theta   = np.imag(result) * conv   # deg/mm

    out_dir = os.path.dirname(file_xyz) or "."

    # ── Write data file ────────────────────────────────────────────────────────
    data_file = os.path.join(out_dir, f"{seedname}-rho_theta.dat")
    header = (
        f"Seedname: {seedname}\n"
        "# rho + i*theta = (omega^2 / 6*c^2*eps0) * [A^xyz - A^yxz]\n"
        "# Column 1: hbar*omega (eV)\n"
        "# Column 2: rho        (deg/mm)\n"
        "# Column 3: theta      (deg/mm)"
    )
    np.savetxt(data_file,
               np.column_stack([hw_eV, rho, theta]),
               fmt="%24.14e",
               header=header)
    print(f"\n  Output   : {data_file}")

    # ── Plot ───────────────────────────────────────────────────────────────────
    make_plot(seedname, hw_eV, rho, theta, out_dir)

    # ── Terminal preview ───────────────────────────────────────────────────────
    print(f"\n  {'hbar*omega (eV)':>16}  {'rho (deg/mm)':>20}  {'theta (deg/mm)':>20}")
    print(f"  {'─'*16}  {'─'*20}  {'─'*20}")
    step = max(1, len(hw_eV) // 8)
    for i in range(0, len(hw_eV), step):
        print(f"  {hw_eV[i]:16.6f}  {rho[i]:20.10e}  {theta[i]:20.10e}")


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == "__main__":
    pairs = resolve_inputs(sys.argv)

    if not pairs:
        print(
            "No matching file pairs found.\n"
            "Expected files of the form:\n"
            f"    {{seedname}}{SUFFIX_XYZ}\n"
            f"    {{seedname}}{SUFFIX_YXZ}\n"
            "in the current (or specified) directory."
        )
        sys.exit(1)

    print(f"Found {len(pairs)} file pair(s) to process.")
    for seedname, xyz_path, yxz_path in pairs:
        compute_rho_theta(seedname, xyz_path, yxz_path)

    print(f"\n{'─'*60}")
    print("Done.")