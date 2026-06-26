# Multipolar_polarizabilities

A workflow for calculating multipolar (dipole-dipole, dipole–quadrupole, quadrupole–quadrupole, dipole–octupole) polarizabilities
from Wannier functions, and for evaluating the natural optical activity (rotatory power and ellipticity) of materials.
----------------------------------------------------------------------------------------------------------------------------------

This code is provided for the paper: "Multipolar polarizabilities in materials: ab initio calculations with Wannier functions".

                                     [link to be added upon publication]

It extends the Wannier-basis methodology of the vdW-WanMBD code:

                                     https://www.sciencedirect.com/science/article/pii/S0010465525000281

Authors: Diem Thi-Xuan Dang, Dai-Nam Le, and Lilia Woods.

Address: Advanced Materials and Devices Theory Group, Department of Physics, University of South Florida, Tampa, FL, USA.

         https://www.amd-woods-group.com/

------------------------------------------------------------------------------------------------------------------------------------

OVERVIEW:

This package contains four codes. Together with a preparatory DFT / Wannierization step, they form
the following workflow:

  Stage 0 - Electronic structure and maximally localized Wannier functions  (Quantum ESPRESSO + Wannier90)
            (no code provided here; uses pw.x and wannier90.x)

  Stage 1 - Multipolar polarizabilities at imaginary frequency  (Wannier90 / postw90)
            berry_multipolar_polarizabilities.F90
            postw90_multipolar_polarizabilities.x

  Stage 2 - Analytic continuation from imaginary to real frequency  (Pade approximant)
            Pade_approximant_convert_imaginary_to_real_frequencies.py

  Stage 3 - Rotatory power and ellipticity from real-frequency polarizabilities
            compute_rotatory.py

The output of each stage is the input of the next, as illustrated below:

   QE + wannier90  -->  postw90  -->  *-polarizability-dd_ij.dat       -->  Pade  -->  *_real.dat  -->  compute_rotatory
                                      *-polarizability-dq_ijk.dat
                                      *-polarizability-qq_ijkl.dat
                                      *-polarizability-do_ijkl.dat
   (MLWFs)                            (imaginary frequency)                  (real frequency)        (rotatory power, ellipticity)

------------------------------------------------------------------------------------------------------------------------------------

DEPENDENCIES:

- Stage 1: a working Wannier90 / postw90 build (the executable postw90_multipolar_polarizabilities.x
           is provided pre-compiled for x86-64 GNU/Linux). To rebuild the code, rename berry_multipolar_polarizabilities.F90 
           to berry.F90. Replace the original src/postw90/berry.F90 file in the Wannier90 source tree with the renamed file,
           then recompile postw90.
- Stages 2 and 3: python3 with Numpy, Scipy and Matplotlib.

------------------------------------------------------------------------------------------------------------------------------------
==
STAGE 0 :  Electronic structure and maximally localized Wannier functions  (Quantum ESPRESSO + Wannier90)
==
This preparatory stage produces the DFT electronic structure and the maximally localized Wannier functions (MLWFs) 
that all later stages rely on. No code is distributed here; it uses the standard Quantum ESPRESSO and Wannier90 executables.

Usage Notes:

• Obtain the electronic structure of the system using the Quantum ESPRESSO code (pw.x).

• Obtain the MLWFs for the system by interpolating the DFT calculation with wannier90.x.

• These steps yield, in particular, the 'seedname.win' / 'seedname.chk' and related Wannier90 files   required to run Stage 1.

STAGE 1 :  berry_multipolar_polarizabilities.F90  /  postw90_multipolar_polarizabilities.x

berry_multipolar_polarizabilities.F90 is a modified version of the Wannier90 module src/postw90/berry.F90.
In addition to the standard "Berry phase" quantities, it computes the optical polarizabilities of the system at imaginary frequency
from the maximally localized Wannier functions:

  - dipole–dipole           (rank 2)
  - dipole–quadrupole       (rank 3)
  - quadrupole–quadrupole   (rank 4)
  - dipole–octupole         (rank 4)

postw90_multipolar_polarizabilities.x is the corresponding pre-compiled postw90 executable.

Usage Notes:

• Request the polarizability evaluation by including the Kubo task in the postw90 input file 'seedname.win':

        berry        = true
        berry_task   = kubo
        kubo_freq_min, kubo_freq_max, kubo_freq_step   (define the imaginary-frequency grid)

  (the polarizability evaluation reuses the Kubo frequency grid; it is activated together with berry_task = kubo.)

• Copy postw90_multipolar_polarizabilities.x to the working folder and execute:

        ./postw90_multipolar_polarizabilities.x seedname

• This generates the following output files (the achar indices i, j, k, l run over x, y, z):

        'seedname'-polarizability-dd_ij.dat      (9 files)   dipole–dipole           [Farad/Angstrom]
        'seedname'-polarizability-dq_ijk.dat     (27 files)  dipole–quadrupole       [Farad]
        'seedname'-polarizability-qq_ijkl.dat    (81 files)  quadrupole–quadrupole   [Farad·Angstrom]
        'seedname'-polarizability-do_ijkl.dat    (81 files)  dipole–octupole         [Farad·Angstrom]

  Each file has the format:

        Column 1 : hbar*xi (imaginary frequency)   [eV]
        Column 2 : Re(polarizability)
        Column 3 : Im(polarizability)

STAGE 2 :  Pade_approximant_convert_imaginary_to_real_frequencies.py

Analytic continuation of the polarizabilities from the imaginary-frequency axis f(i*xi) to the real-frequency axis 
f(omega + i*eta), using a self-consistent two-point (Thiele) Pade approximant.
For each input file f(0) and the high-frequency tail coefficient are determined by leave-one-out cross-validation on the 
imaginary axis, which stabilizes the continuation.

Usage Notes:

• Copy the imaginary-frequency polarizability files produced in Stage 1 into a working folder
  (typically the dipole–quadrupole files dq_xyz and dq_yxz are needed for optical activity, but any
  of the dd / dq / qq / do files may be continued).

• Execute, for example:

        python Pade_approximant_convert_imaginary_to_real_frequencies.py .

• Command-line options:

        folder              directory containing the input .dat files             (default: current directory)
        --pattern   STR     glob pattern for input files                          (default: *.dat)
        --suffix    STR     suffix appended to each output filename               (default: _real)
        --eta       FLOAT   Lorentzian broadening of the real axis, in eV         (default: 0.001)
        --w-min     FLOAT   lower bound of the real-frequency grid, in eV         (default: 0.0)
        --w-max     FLOAT   upper bound of the real-frequency grid, in eV         (default: 1.0)
        --dw        FLOAT   step size of the real-frequency grid, in eV           (default: 0.01)

  Files whose name already contains the suffix (e.g. '_real') are skipped, so the script can be
  rerun safely in the same folder.

• For every input file 'stem.dat' the script writes:

        'stem'_real.dat          continued data; columns: real frequency (eV), Re f(omega), Im f(omega)
        figs/'stem'.jpg          plot of Re f, Im f on the real axis together with f(i*omega)

  The real-frequency dipole–quadrupole files 'seedname'-polarizability-dq_xyz_real.dat and
  'seedname'-polarizability-dq_yxz_real.dat are the inputs required by Stage 3.

STAGE 3 :  compute_rotatory.py

Computes the natural optical activity of the material from the real-frequency dipole–quadrupole polarizability, following:

        rho(omega) + i*theta(omega) = ( omega^2 / 6*c^2*eps0 ) * [ A^xyz(omega) - A^yxz(omega) ]

  where  rho(omega)   = rotatory power   (real part)        [deg/mm]
         theta(omega) = ellipticity      (imaginary part)   [deg/mm]
         A^xyz, A^yxz = mixed dipole–quadrupole polarizability tensors   [Farad]

Usage Notes:

• Place the two real-frequency dipole–quadrupole files produced in Stage 2 in a working folder:

        'seedname'-polarizability-dq_xyz_real.dat
        'seedname'-polarizability-dq_yxz_real.dat

  Each file is expected to have:

        Column 1 : hbar*omega   [eV]
        Column 2 : Re(A)        [F]
        Column 3 : Im(A)        [F]   (optional; treated as zero if absent)

• Execute in one of the following ways:

        python compute_rotatory.py                      auto-detect pairs in the current directory
        python compute_rotatory.py /path/to/dir         auto-detect pairs in the given directory
        python compute_rotatory.py xyz.dat yxz.dat      use the two explicit file paths

  The script auto-detects file pairs by matching the '-polarizability-dq_xyz_real.dat' suffix and
  confirming that the corresponding '-polarizability-dq_yxz_real.dat' partner exists. It also checks
  that the two files share the same energy axis.

• For each seedname the script writes:

        'seedname'-rho_theta.dat     columns: hbar*omega (eV), rho (deg/mm), theta (deg/mm)
        'seedname'-rho_theta.png     plot of rho (red) and theta (blue) versus hbar*omega

------------------------------------------------------------------------------------------------------------------------------------

Examples:

• Multipolar polarizabilities (Stages 1 and 2) are included for six representative bulk layered materials:

      examples/graphite
      examples/bulk hBN
      examples/bulk MoS2
      examples/bulk 2H-WTe2
      examples/bulk Td-WTe2
      examples/bulk Te

• Rotatory power and ellipticity (Stage 3) are included for two materials:

      examples/bulk Td-WTe2
      examples/bulk Te

------------------------------------------------------------------------------------------------------------------------------------

Acknowledgements:

D.T-X.D. acknowledges financial support from the Presidential Doctoral Fellowship sponsored by the University of South Florida. 
L.M.W. acknowledges financial support from the US Department of Energy under Grant No. DE-FG02-06ER46297.
The authors used Claude (with Sonnet 4.6) and M365 Copilot Enterprise (with GPT-5.5) to debug and optimize code performance.

------------------------------------------------------------------------------------------------------------------------------------

For assistance with running please email Diem Thi-Xuan Dang at dangt1@usf.edu.

------------------------------------------------------------------------------------------------------------------------------------

berry_multipolar_polarizabilities.F90 is distributed as part of the Wannier90 code and is released under
the GNU General Public License ver. 3. The accompanying Python codes are likewise released under the
GNU General Public License ver. 3. Please consult the included LICENSE file for detailed licensing conditions.
