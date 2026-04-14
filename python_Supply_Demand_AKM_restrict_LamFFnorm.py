
# =============================================================================
# Identifying Relationship-level Effects Using covariance restrictions: De Jonghe and Lewis (2026)
# =============================================================================
# The model assumes that observed price (P) and quantity (Q) changes are
# linear combinations of an underlying supply shock and demand shock,
# governed by a 2x2 mixing matrix A:
#   [dP]   =  A  @  [supply shock]
#   [dQ]            [demand shock]
#
#
# Inputs:
#   csv_FBT_panel_QP_long.csv  — long-format panel with columns:
#                                Firm, Bank, Time, dQdh (quantity change), dP (price change)
#
# Outputs:
#   csv_FBT_panel_QPSD_long.csv  — input data augmented with Supply and Demand columns
#   summary_output.csv           — one row of results per run, appended across runs
#
# =============================================================================


# %%
# -----------------------------------------------------------------------------
# Section 1: Imports and Setup
# -----------------------------------------------------------------------------
# Standard scientific Python libraries:
#   pandas     — tabular data handling
#   numpy      — numerical arrays and linear algebra
#   scipy      — advanced linear algebra (block_diag) and optimisation (minimize)
#   joblib     — parallel execution across CPU cores
#   os/gc/time — file system, memory management, and timing utilities

import pandas as pd
import numpy as np
import scipy.linalg as slin
from scipy.optimize import minimize
from scipy.stats import chi2
from scipy.linalg import block_diag
from joblib import Parallel, delayed
import os
import gc
import time

# =============================================================================
# USER CONFIGURATION — edit this section before running
# =============================================================================
#
# 1. STUDY IDENTIFIERS
#    These are written as-is into the summary output CSV to identify the run.
#    They do not affect the estimation; they are purely for traceability across runs.
#    Use os.getenv() when using a shell command calling the python script in stata (for a loop)
#    Can be replaced with specific values for users using the script on a standalone one-run basis

country         = os.getenv("COUNTRY")          # country code, e.g. 'AT', 'BE', 'DE', 'IT', ...
start_period    = os.getenv("START_PERIOD")     # start of the sample window, e.g. YYYYQQ or YYYYMM
end_period      = os.getenv("END_PERIOD")       # end of the sample window, e.g. YYYYQQ or YYYYMM

# 2. INPUT FILE
#    Long-format panel CSV with columns: [Firm, Bank, Time, dQdh, dP].
#    Must be in the same folder as this script, or provide an absolute path.
input_file = "csv_FBT_panel_QP_long.csv"

# 3. OUTPUT FILES
#    - The shock output is always written fresh (overwritten on each run).
#    - The summary is appended: one row per run, header written on first run only.
output_shocks_file  = "csv_FBT_panel_QPSD_long.csv"
output_summary_file = "summary_output.csv"

# 4. PARALLELISATION
#    Number of CPU cores to use in the bank-loop parallelisation.
#    Set to -1 to use all available cores (recommended).
#    Set to 1 to disable parallelisation entirely (useful for debugging).
#    On shared servers, set to a specific number to be a good citizen.
n_jobs = -1

# 5. RANDOM SEED
#    The AKM test uses random starting values across nstarts optimisations.
#    Fix this seed for reproducibility. Set to None to disable.
random_seed = 42

# 6. AKM TEST SETTINGS
#    nstarts: number of random restarts for the overidentification test.
#             Higher = more reliable minimum but slower. 100 is usually sufficient.
#    significance_level: for the chi-squared rejection threshold (e.g. 0.10 = 10%).
nstarts           = 100
significance_level = 0.10

# =============================================================================
# END OF USER CONFIGURATION — do not edit below this line unless you know
# what you are doing
# =============================================================================

notebook_start_time = time.time()
print("Starting notebook execution...")

print(f"Running for {country} from {start_period} to {end_period}")

# Set the working directory to the folder where this script is saved,
# so that relative file paths (e.g. CSV files) work on any machine.
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Load the CSV file
df = pd.read_csv(input_file)

# Rename columns for consistency. The raw CSV contains dQdh (quantity change)
# and dP (price change); these are renamed to Q and P throughout the script.
df.columns = ['Firm', 'Bank', 'Time', 'Q', 'P']

# Create ID-to-index mappings.
# The raw Firm, Bank and Time identifiers can be arbitrary integers or strings.
# We map each unique ID to a contiguous integer index (0, 1, 2, ...) so they
# can be used directly as array positions.
firm_ids = df['Firm'].unique()
bank_ids = df['Bank'].unique()
time_ids = df['Time'].unique()

firm_index = {id_: i for i, id_ in enumerate(firm_ids)}
bank_index = {id_: i for i, id_ in enumerate(bank_ids)}
time_index = {id_: i for i, id_ in enumerate(time_ids)}

# Dimensions: F = number of firms, B = number of banks, T = number of periods.
F, B, T = len(firm_ids), len(bank_ids), len(time_ids)

# Map IDs to indices and store in the dataframe.
df['f_idx'] = df['Firm'].map(firm_index)
df['b_idx'] = df['Bank'].map(bank_index)
df['t_idx'] = df['Time'].map(time_index)

# Initialize 3D arrays of shape (F, B, T), filled with NaN to denote missing
# observations. Not every firm-bank-time combination will have data.
# etap holds price changes (P), etaq holds quantity changes (Q).
etap = np.full((F, B, T), np.nan)
etaq = np.full((F, B, T), np.nan)

# %%
# -----------------------------------------------------------------------------
# Section 2: Fill 3D Arrays
# -----------------------------------------------------------------------------
# Convert the long-format dataframe into 3D NumPy arrays (F x B x T).
# Each entry [f, b, t] holds the observed P or Q for firm f, bank b, at time t.
# Using advanced (integer) indexing allows filling all entries in a single step
# without looping, which is much faster than iterating row by row.

# Convert index columns to integer arrays for use as array positions.
f_idx = df['f_idx'].astype(int).to_numpy()
b_idx = df['b_idx'].astype(int).to_numpy()
t_idx = df['t_idx'].astype(int).to_numpy()

# Convert data columns to NumPy arrays.
P_vals = df['P'].to_numpy()
Q_vals = df['Q'].to_numpy()

# Fill arrays using advanced indexing: each row of the dataframe maps to
# exactly one cell in the 3D array.
etap[f_idx, b_idx, t_idx] = P_vals
etaq[f_idx, b_idx, t_idx] = Q_vals

# Free intermediate variables no longer needed to reduce memory usage.
del firm_ids, bank_ids, time_ids
del firm_index, bank_index, time_index
del F, B, T
del f_idx, b_idx, t_idx, P_vals, Q_vals

gc.collect()

# %%
# -----------------------------------------------------------------------------
# Section 3: Count Pair Combinations and Replace NaN with Zero
# -----------------------------------------------------------------------------
# For the moment estimator, we need to count how many firms each bank
# serves (Fb) and how many banks each firm uses (Bf), at each time period.
#
# NFF is the total number of firm pairs sharing at least one bank (across all
# banks and periods). NBB is the analogous count for bank pairs sharing a firm.
# These serve as normalisation constants in the moment estimators.
#
# NaN entries (missing observations) are then replaced with 0 so that
# subsequent matrix operations treat unobserved cells as non-contributing.

F, B, T = etaq.shape

NFF = 0
NBB = 0

# Fbstore[t, b]: number of firms observed at bank b in period t.
# Bfstore[t, f]: number of banks observed for firm f in period t.
Fbstore = np.full((T, B), np.nan)
Bfstore = np.full((T, F), np.nan)

for t in range(T):
    # Count non-NaN entries along the firm axis (axis=0) for each bank.
    Fb = np.sum(~np.isnan(etaq[:, :, t]), axis=0)
    Fbstore[t, :] = Fb
    Fb = Fb.astype(np.int64)

    # Count non-NaN entries along the bank axis (axis=1) for each firm.
    Bf = np.sum(~np.isnan(etap[:, :, t]), axis=1)
    Bfstore[t, :] = Bf.T

    # Number of firm pairs at each bank: Fb*(Fb-1)/2, summed across banks.
    NFF += np.sum(np.multiply(Fb, (Fb - 1))) / 2

    # Number of bank pairs at each firm: Bf*(Bf-1)/2, summed across firms.
    NBB += np.sum(np.multiply(Bf, (Bf - 1))) / 2

    # Replace NaN with 0 for subsequent matrix algebra.
    eq = etaq[:, :, t]
    eq[np.isnan(eq)] = 0
    etaq[:, :, t] = eq

    ep = etap[:, :, t]
    ep[np.isnan(ep)] = 0
    etap[:, :, t] = ep

# Memory cleanup: Fb and Bf are loop variables no longer needed.
del Fb, Bf
gc.collect()


# %%
# -----------------------------------------------------------------------------
# Section 4: Moment Computation Worker Function
# -----------------------------------------------------------------------------
# For each bank b at a given time period, this function computes:
#
#   vft_col: a 3-element vector of firm-level second moments contributed by
#            bank b. These are the sum of cross-products of P and Q across
#            all pairs of firms that share bank b:
#               [sum_pairs(P_f * P_f'), sum_pairs(P_f * Q_f'), sum_pairs(Q_f * Q_f')]
#            The "pairs" structure (i.e. dividing by 2 and using off-diagonal
#            products) ensures each pair is counted once.
#
#   vbt_part: an (F x 3) matrix of bank-level second moments contributed by
#             bank b. Row f contains the cross-products of firm f's (P, Q)
#             with all other banks' (P, Q) at the same firm. This captures
#             the covariance structure across banks within each firm.
#
# The function is designed to be called in parallel across all banks b,
# which is why it receives the full ep and eq slices as arguments.

def process_b(b, ep, eq):
    # Extract the price and quantity vectors for bank b across all firms.
    ep_b = ep[:, b]   # shape (F,)
    eq_b = eq[:, b]   # shape (F,)

    # Compute scalar summaries for bank b.
    ep_b_sum = np.sum(ep_b)
    eq_b_sum = np.sum(eq_b)
    ep_b_sq_sum = np.dot(ep_b, ep_b)
    ep_b_eq_b_sum = np.dot(ep_b, eq_b)
    eq_b_sq_sum = np.dot(eq_b, eq_b)

    # Firm-level moment vector for bank b.
    # Using the identity: sum_{f<f'} x_f * x_f' = [(sum x_f)^2 - sum x_f^2] / 2
    # This avoids an explicit double loop over firm pairs.
    vft_col = np.array([
        (ep_b_sum**2 - ep_b_sq_sum) / 2,          # PP moment
        (ep_b_sum * eq_b_sum - ep_b_eq_b_sum) / 2, # PQ moment
        (eq_b_sum**2 - eq_b_sq_sum) / 2            # QQ moment
    ])

    # Bank-level moment matrix: for each firm f, accumulate cross-products
    # of (ep_b, eq_b) with all other banks' columns (ep_{b'}, eq_{b'}).
    # Split into banks before b and banks after b to avoid double-counting.
    vbt_part = np.zeros((ep.shape[0], 3))

    if b > 0:
        ep_rest1 = ep[:, :b]   # all banks with index < b
        eq_rest1 = eq[:, :b]
        # einsum 'i,ij->i': for each firm i, compute ep_b[i] * ep_rest1[i, :]
        # summed over the j (bank) dimension, giving a scalar per firm.
        vbt_part[:, 0] += np.einsum('i,ij->i', ep_b, ep_rest1) / 2  # PP
        vbt_part[:, 1] += np.einsum('i,ij->i', ep_b, eq_rest1) / 2  # PQ
        vbt_part[:, 2] += np.einsum('i,ij->i', eq_b, eq_rest1) / 2  # QQ

    if b < ep.shape[1] - 1:
        ep_rest2 = ep[:, b+1:]  # all banks with index > b
        eq_rest2 = eq[:, b+1:]
        vbt_part[:, 0] += np.einsum('i,ij->i', ep_b, ep_rest2) / 2
        vbt_part[:, 1] += np.einsum('i,ij->i', ep_b, eq_rest2) / 2
        vbt_part[:, 2] += np.einsum('i,ij->i', eq_b, eq_rest2) / 2

    return b, vft_col, vbt_part


# %%
# -----------------------------------------------------------------------------
# Section 5: Parallelised Moment Accumulation
# -----------------------------------------------------------------------------
# For each time period t, call process_b() in parallel across all banks b.
# Results are accumulated into two global arrays:
#
#   vf: shape (3, B*T) — firm-level second moments, stacked across periods.
#       Columns t*B to (t+1)*B correspond to period t.
#
#   vb: shape (T*F, 3) — bank-level second moments, stacked across periods.
#       Rows t*F to (t+1)*F correspond to period t.
#
# These arrays are later used to form the estimated moment matrices fmat and
# bmat (firm- and bank-level second moment matrices of the mixing matrix A).

# Pre-allocate output arrays.
vf = np.zeros((3, B * T))
vb = np.zeros((T * F, 3))

for t in range(T):
    ep = etap[:, :, t]  # Price slice for period t: shape (F, B)
    eq = etaq[:, :, t]  # Quantity slice for period t: shape (F, B)
    print(f"Processing t = {t}")

    # Parallelise across banks using the 'loky' multiprocessing backend.
    # n_jobs is set on top in the user-configuration setting. 
    # E.g. n_jobs=8 uses up to 8 CPU cores simultaneously.
    # If working with memory-mapped arrays, 'threading' may be faster.
    results = Parallel(n_jobs=n_jobs, backend='loky')(
        delayed(process_b)(b, ep, eq) for b in range(B)
    )
    # results = Parallel(n_jobs=n_jobs, backend='threading')(delayed(process_b)(b, ep, eq) for b in range(B))  # Try this if memmap

    # Assemble results from all banks into period-t matrices.
    vft = np.zeros((3, B))
    vbt = np.zeros((F, 3))

    for b, vft_col, vbt_part in results:
        vft[:, b] = vft_col   # Store the 3-vector for bank b.
        vbt += vbt_part        # Accumulate firm-level cross-bank moments.

    # Store period t results in the global arrays.
    vf[:, t * B:(t + 1) * B] = vft
    vb[t * F:(t + 1) * F, :] = vbt


# Memory cleanup: large intermediate arrays no longer needed.
del eq, ep, results, vft, vbt, etap, etaq
gc.collect()

# %%
# -----------------------------------------------------------------------------
# Section 6: Estimate the Elasticity Matrix A via Eigendecomposition
# -----------------------------------------------------------------------------
# The De Jonghe-Lewis (2026) identification result states that:
#
#   fmat = A @ LamFF @ A.T    (firm-level second moment matrix)
#   bmat = A @ LamBB @ A.T    (bank-level second moment matrix)
#
# where LamFF and LamBB are diagonal matrices of the shock variances as seen
# from the firm and bank dimensions respectively. Under the normalisation used
# here (LamFF = I), the ratio fmat @ inv(bmat) has eigenvalues equal to the
# diagonal elements of LamFF @ inv(LamBB), and its eigenvectors are the
# columns of A. We recover A as the matrix of right eigenvectors of this ratio.
#
# fmat and bmat are 2x2 symmetric matrices with entries:
#   [E(PP), E(PQ)]
#   [E(PQ), E(QQ)]
# estimated separately from the firm-pair (vf) and bank-pair (vb) moments.

# Bank-level second moment matrix (bmat): average over all bank pairs.
vbb = np.sum(vb, axis=0) / NBB
bmat = np.array([[vbb[0], vbb[1]],
                 [vbb[1], vbb[2]]])
vbb = vbb.reshape(1, -1)

# Firm-level second moment matrix (fmat): average over all firm pairs.
vfb = np.sum(vf, axis=1) / NFF
fmat = np.array([[vfb[0], vfb[1]],
                 [vfb[1], vfb[2]]])
vfb = vfb.reshape(-1, 1)

# Form the ratio matrix whose eigenstructure identifies A.
ratio = fmat @ np.linalg.inv(bmat)

# Eigendecomposition: Lamhat contains eigenvalues, Ahat contains right
# eigenvectors as columns. These eigenvectors are the initial estimate of A.
Lamhat, Ahat = np.linalg.eig(ratio)  # Ahat is right eigenvectors, Lamhat are eigenvalues (not used)
# _, Ahat2 = np.linalg.eig(ratio.T)  # Ahat2 is left eigenvectors (not used)
# Lamhat = np.diag(Lamhat)


# %%
# -----------------------------------------------------------------------------
# Section 7: Inference — Sandwich Variance Estimator
# -----------------------------------------------------------------------------
# We compute asymptotic standard errors for the elements of A using a
# sandwich (delta method) variance estimator. The score contributions for
# each firm pair (vf) and bank pair (vb) are recentred by subtracting the
# estimated mean, then used to form heteroskedasticity-robust variance
# matrices WFF and WBB.
#
# Two versions are computed:
#   WFF / WBB   — standard sandwich estimator (pairs as i.i.d. units)
#   WFF_R / WBB_R — time-robust version, which aggregates scores within
#                   each period before forming the outer product. This is
#                   more conservative and robust to within-period correlation.

# Recentre firm-level scores: subtract the weighted mean contribution.
# wf[t, b] = Fb*(Fb-1)/2 is the number of firm pairs at bank b in period t.
wf = np.reshape(Fbstore, (1, T * B))
wf = wf * (wf - 1) / 2
wb = np.reshape(Bfstore, (T * F, 1))
wb = wb * (wb - 1) / 2

vf = vf - wf * vfb   # Recentred firm-level scores
vb = vb - wb * vbb   # Recentred bank-level scores

# Standard sandwich variance matrices.
WFF = 1 / NFF**2 * vf @ vf.T   # shape (3, 3)
WBB = 1 / NBB**2 * vb.T @ vb   # shape (3, 3)

# Time-robust variance: sum scores within each period first, then take
# the outer product. This accounts for within-period dependence.
vfT = 0
vbT = 0
for t in range(T):
    vfT += vf[:, t * B:(t + 1) * B]
    vbT += vb[t * F:(t + 1) * F, :]

WFF_R = 1 / NFF**2 * vfT @ vfT.T
WBB_R = 1 / NBB**2 * vbT.T @ vbT

# Normalise Ahat so that the diagonal of invAhat @ fmat @ invAhat.T equals 1.
# This sets the scale of each column of A so that the firm-level shock variance
# (LamFF) is normalised to the identity matrix, as used in the paper
invAhat = np.linalg.inv(Ahat)
Ahat = Ahat @ np.diag(np.lib.scimath.sqrt(np.diag(invAhat @ fmat @ invAhat.T)))


# %%
# -----------------------------------------------------------------------------
# Section 8: Auxiliary Functions
# -----------------------------------------------------------------------------

def labeld(Ahat, A):
    """
    Resolve sign and column-order ambiguity in the estimated mixing matrix.

    Eigenvectors are unique only up to sign flips and column permutations.
    This function tries all 8 valid combinations (4 sign patterns x 2 column
    orders) and returns the version closest to the target matrix A (measured
    by squared Frobenius norm). The target A is built from the data's standard
    deviations and encodes the expected sign structure of supply and demand.

    Parameters
    ----------
    Ahat : (2, 2) ndarray — estimated elasticity matrix (up to sign/order ambiguity)
    A    : (2, 2) ndarray — target matrix encoding the desired sign convention

    Returns
    -------
    Alab : (2, 2) ndarray — labelled version of Ahat closest to A
    """
    # sca defines 4 sign patterns applied column-wise:
    #   [+1, +1]: keep both columns as-is
    #   [-1, -1]: flip both columns
    #   [+1, -1]: keep col 1, flip col 2
    #   [-1, +1]: flip col 1, keep col 2
    sca = np.array([[1, 1], [-1, -1], [1, -1], [-1, 1]])
    norms = []

    for i in range(4):          # loop over sign patterns
        for j in range(2):      # j=0: original column order, j=1: columns swapped
            if j == 0:
                diff = Ahat * sca[i] - A
            else:
                diff = Ahat[:, [1, 0]] * sca[i] - A
            norms.append(np.sum(diff ** 2))

    # ind runs from 0 to 7. ind = i*2 + j, so:
    #   i = ind // 2  (which sign pattern)
    #   j = ind % 2   (original or swapped columns)
    ind = np.argmin(norms)

    if (ind + 1) % 2 == 0:
        # Even ind+1 means j=1: columns were swapped in the winning candidate.
        Alab = Ahat[:, [1, 0]] * sca[ind // 2]
    else:
        # Odd ind+1 means j=0: original column order in the winning candidate.
        Alab = Ahat * sca[ind // 2]

    return Alab


def jacd(theta):
    """
    Compute the Jacobian matrix of the moment conditions with respect to theta.

    theta = [A11, A21, A12, A22, LamBB11, LamBB22] (6 parameters).
    The moment conditions are the 6 unique elements of fmat and bmat expressed
    as functions of A and the Lambda matrices. The Jacobian Phi is used in the
    delta method to propagate variance from the moment estimators to the
    structural parameter estimates.

    Parameters
    ----------
    theta : array of length 6 — structural parameters

    Returns
    -------
    Phi : (6, 6) ndarray — Jacobian of moment conditions w.r.t. theta
    """
    Phi = np.array([
        [2*theta[0],            0,  2*theta[2],            0,           0,           0],
        [theta[1],      theta[0],   theta[3],      theta[2],           0,           0],
        [0,         2*theta[1],           0,    2*theta[3],           0,           0],
        [2*theta[0]*theta[4],   0,  2*theta[2]*theta[5],   0,  theta[0]**2,  theta[2]**2],
        [theta[1]*theta[4], theta[0]*theta[4], theta[3]*theta[5], theta[2]*theta[5], theta[0]*theta[1], theta[2]*theta[3]],
        [0,  2*theta[1]*theta[4],           0, 2*theta[3]*theta[5],  theta[1]**2,  theta[3]**2]
    ])
    return Phi


def check_complex(matrix):
    """
    Guard against complex-valued eigenvectors in small samples.

    In datasets with very few banks, the ratio matrix fmat @ inv(bmat) may
    have complex eigenvalues, causing Ahat to be complex. This is a sign that
    the data do not support identification (too little variation). In that
    case, a neutral fallback matrix [[1,-1],[1,1]] is returned, which encodes
    a generic sign convention without making strong assumptions.

    Parameters
    ----------
    matrix : (2, 2) array — estimated Ahat after normalisation

    Returns
    -------
    (2, 2) real ndarray — original matrix if real, fallback if complex
    """
    mat = np.array(matrix)

    if mat.shape != (2, 2):
        raise ValueError("Matrix must be 2x2")

    has_complex = np.any(np.imag(mat) != 0)

    if has_complex:
        print("Complex elements detected - returning fallback matrix [[1,-1],[1,1]]")
        return np.array([[1, -1],
                         [1,  1]])
    else:
        print("All elements are real - returning original matrix")
        return np.real(mat)


# %%
# -----------------------------------------------------------------------------
# Section 9: Label Ahat and Compute Final Standard Errors
# -----------------------------------------------------------------------------
# After eigendecomposition, A is only identified up to sign flips and column
# permutations. labeld() resolves this by selecting the orientation of Ahat
# that most closely matches a target built from the data's standard deviations.
# The target encodes the prior that the supply column should have opposite
# signs for P and Q (a downward-sloping demand curve), while the demand column
# has the same sign for both.
#
# Once Ahat is fully determined, we recover:
#   invAhat   — inverse of A, used to compute Supply and Demand residuals
#   Lam2      — diagonal of invAhat @ bmat @ invAhat.T, i.e. the bank-level
#               shock variances (LamBB) implied by the estimated A
#   thetahat  — full parameter vector [A11, A21, A12, A22, LamBB11, LamBB22]
#   SEs       — standard errors via the delta method (standard sandwich)
#   SEs_R     — standard errors via the delta method (time-robust sandwich)

# Replace Ahat with fallback if eigenvectors are complex (small-sample case).
Ahat = check_complex(Ahat)

# Resolve sign and column-order ambiguity using the data's standard deviations
# as a reference. Target structure: col1=[+std(P), +std(Q)] (demand),
# col2=[-std(P), +std(Q)] (supply)).
Ahat = labeld(Ahat, np.array([[df['P'].std(), -df['P'].std()],
                               [df['Q'].std(),  df['Q'].std()]]))

# Recover the inverse and implied shock variances.
invAhat = np.linalg.inv(Ahat)
Lam2 = np.diag(invAhat @ bmat @ invAhat.T).reshape(-1, 1)

# Stack all parameters into a single vector for the delta method.
# Note: Ahat.T reshaped row-major gives [A11, A21, A12, A22].
thetahat = np.vstack((np.reshape(Ahat.T, (4, 1)), Lam2))  # shape (6, 1)
                                                            # Note: reshape order differs from MATLAB

# Compute the Jacobian of the moment conditions at the estimated parameters.
Phi = jacd(thetahat[:, 0])
invPhi = np.linalg.inv(Phi)

# Delta method: propagate variance of moment estimators to parameter estimates.
# Omega = invPhi @ block_diag(WFF, WBB) @ invPhi.T
# The top-left 4x4 block corresponds to the A matrix elements.
Omega = invPhi @ slin.block_diag(WFF, WBB) @ invPhi.T
SEs = np.sqrt(np.diag(Omega[0:4, 0:4]))

Omega_R = invPhi @ slin.block_diag(WFF_R, WBB_R) @ invPhi.T
SEs_R = np.sqrt(np.diag(Omega_R[0:4, 0:4]))


# %%
# -----------------------------------------------------------------------------
# Section 10: AKM Overidentification Test
# -----------------------------------------------------------------------------
# The model imposes restrictions on fmat and bmat: both must be expressible
# as A @ Lam @ A.T for the same A but different diagonal Lambda matrices.
# With LamFF restricted to the identity (normalisation) and LamBB diagonal,
# the model is exactly identified in general. However, the AKM model implies two additional
# zero restrictions on LamFF and LamBB, yielding 2 overidentifying restrictions.
#
# AKMtest() computes the minimum distance objective that measures how well
# the estimated A fits both fmat and bmat simultaneously. Under the null that
# the model is correctly specified, this statistic is chi-squared with 2 d.f.
#
# Because the objective is non-convex, we run the optimisation from 100
# random starting points and take the minimum, to avoid local minima.

def AKMtest(theta, fmat, bmat, What):
    """
    Compute the AKM minimum distance test statistic.

    Measures how well the 2x2 matrix A (encoded in theta) jointly fits
    the firm-level (fmat) and bank-level (bmat) second moment matrices under
    the restrictions LamFF = [[0,0],[0,1]] and LamBB = [[1,0],[0,0]].

    The fit is measured by the weighted squared distance between the implied
    moments and the estimated moments, using the robust variance matrix What
    as the weight (so that less precisely estimated moments receive less weight).

    Parameters
    ----------
    theta : array of length 4 — elements of A (A11, A21, A12, A22)
    fmat  : (2, 2) ndarray — estimated firm-level second moment matrix
    bmat  : (2, 2) ndarray — estimated bank-level second moment matrix
    What  : (6, 6) ndarray — weighting matrix (block diagonal of WFF_R, WBB_R)

    Returns
    -------
    obj : float — weighted distance (test statistic value at this theta)
    """
    A = theta[:4].reshape(2, 2)          # reconstruct A from parameter vector
    LamFF = np.array([[0, 0], [0, 1]])   # normalisation: LamFF11=0, LamFF22=1
    LamBB = np.array([[1, 0], [0, 0]])   # LamBB11=1, LamBB22=0 (restriction)

    # Residuals: how far are the implied moments from the estimated ones?
    d1 = fmat - A @ LamFF @ A.T
    d2 = bmat - A @ LamBB @ A.T

    # Flatten in column-major order (matching MATLAB convention) and extract
    # the 3 unique elements of each symmetric 2x2 matrix (indices 0, 1, 3).
    d1_flat = d1.T.flatten()
    d2_flat = d2.T.flatten()
    q1 = d1_flat[[0, 1, 3]]   # [d1_11, d1_21, d1_22]
    q2 = d2_flat[[0, 1, 3]]   # [d2_11, d2_21, d2_22]
    q = np.concatenate([q1, q2])  # combined residual vector (length 6)

    # Weighted distance: q.T @ inv(What) @ q.
    # Use solve() instead of explicit inversion for numerical stability.
    try:
        obj = q.T @ np.linalg.solve(What, q)
    except np.linalg.LinAlgError:
        # Fallback to pseudoinverse if What is singular.
        obj = q.T @ np.linalg.pinv(What) @ q

    return obj


# Build starting values for the optimisation from the just-identified estimate.
# Normalise Ahat so its diagonal elements are 1, then back out the implied
# Lambda matrices. This gives a starting point close to the true optimum.
# In MATLAB: Ahat * diag(diag(Ahat))^-1
Ahat_norm = Ahat @ np.linalg.inv(np.diag(np.diag(Ahat)))
Ahat_inv = np.linalg.inv(Ahat_norm)
LamFF = Ahat_inv @ fmat @ Ahat_inv.T
LamBB = Ahat_inv @ bmat @ Ahat_inv.T

# ord1: goodness of the current column ordering — rewards positive Lambda values
# (negative variances would be incoherent with the model).
ord1 = (LamFF[1, 1]**2 * (LamFF[1, 1] > 0) +
        LamBB[0, 0]**2 * (LamBB[0, 0] > 0))

# Also try the column-swapped version and pick whichever gives positive Lambdas.
Ahata = Ahat[:, [1, 0]]
Ahata = Ahata @ np.linalg.inv(np.diag(np.diag(Ahata)))
Ahata_inv = np.linalg.inv(Ahata)
LamFFa = Ahata_inv @ fmat @ Ahata_inv.T
LamBBa = Ahata_inv @ bmat @ Ahata_inv.T
ord2 = (LamFFa[1, 1]**2 * (LamFFa[1, 1] > 0) +
        LamBBa[0, 0]**2 * (LamBBa[0, 0] > 0))

# Scale the starting vector by the square root of the implied Lambda values
# so that the optimiser starts in a region where the constraints are feasible.
if ord1 >= ord2:
    start = Ahat_norm.reshape(-1, order='F')  # column-major reshape, as in MATLAB
    LamFF[1, 1] = max(LamFF[1, 1], 0.01)     # clamp to avoid zero/negative start
    LamBB[0, 0] = max(LamBB[0, 0], 0.01)
    start[2:4] = start[2:4] * np.sqrt(LamFF[1, 1])
    start[0:2] = start[0:2] * np.sqrt(LamBB[0, 0])
else:
    start = Ahata.reshape(-1, order='F')
    LamFFa[1, 1] = max(LamFFa[1, 1], 0.01)
    LamBBa[0, 0] = max(LamBBa[0, 0], 0.01)
    start[2:4] = start[2:4] * np.sqrt(LamFFa[1, 1])
    start[0:2] = start[0:2] * np.sqrt(LamBBa[0, 0])


# Run the minimisation from 'nstarts' random starting points to avoid local minima.
# nstarts (number of simulation is defined in user configuration settings on top)
# Each start perturbs the base starting vector with standard normal noise.
# SLSQP is the closest Python equivalent to MATLAB's fmincon.
# Apply random seed for reproducibility (set in user configuration above).
if random_seed is not None:
    np.random.seed(random_seed)
    
What = block_diag(WFF_R, WBB_R)         # robust weighting matrix (6x6)
options = {'maxiter': 10000}             # equivalent to MaxFunEvals in MATLAB

val = np.zeros(nstarts)
flag = np.zeros(nstarts)

for i in range(nstarts):
    start_vals = start + np.random.randn(4)
    start_vals = np.array(start_vals).flatten()   # ensure 1D
    result = minimize(lambda theta: AKMtest(theta, fmat, bmat, What),
                      start_vals,
                      method='SLSQP',
                      options=options)
    val[i] = result.fun
    flag[i] = result.success

# The test statistic is the minimum objective value across all starting points.
# Reject the null (model is correctly specified) if it exceeds the (user configuration-set)
# percentile of the chi-squared distribution with 2 degrees of freedom.
# Use significance level from user configuration above.
teststat = np.nanmin(val)
rej = teststat > chi2.ppf(1 - significance_level, 2)

print(f"Successful optimizations in AKM test: {np.sum(flag)}/{nstarts}")

# %%
# -----------------------------------------------------------------------------
# Section 11: Timing
# -----------------------------------------------------------------------------
notebook_end_time = time.time()
execution_time = notebook_end_time - notebook_start_time
print(f"Total notebook execution time: {execution_time:.2f} seconds ({execution_time/60:.2f} minutes)")

# %%
# -----------------------------------------------------------------------------
# Section 12: Compute Supply and Demand and Write Output CSV
# -----------------------------------------------------------------------------
# Apply the inverse of the estimated mixing matrix to recover the latent
# supply and demand shocks for each observation:
#
#   [Supply]  =  invAhat  @  [dP]
#   [Demand]                 [dQ]
#
# These are written back into the dataframe and saved to CSV.

df['Demand'] = df['P'] * invAhat[0, 0] + df['Q'] * invAhat[0, 1]
df['Supply'] = df['P'] * invAhat[1, 0] + df['Q'] * invAhat[1, 1]

df[['Firm', 'Bank', 'Time', 'Q', 'P', 'Supply', 'Demand']].to_csv(output_shocks_file, index=False)

# %%
# -----------------------------------------------------------------------------
# Section 13: Write Summary Output
# -----------------------------------------------------------------------------
# Collect all key results into a single-row summary dataframe and append it
# to a running CSV file (one row per country/period run).
#
# Standard error naming convention (se_XY):
#   X = row index of A (1=first row, 2=second row)
#   Y = column index of A (1=supply column, 2=demand column)
# so se12 is the SE of A[0,1] (first row, demand column), etc.
#
# The header is written only if the output file does not yet exist,
# so that multiple runs append cleanly without duplicating the header.

contents_df = {
    'country': country,
    'start': start_period,
    'end': end_period,
    'F': F,
    'B': B,
    'T': T,
    'A11': [Ahat[0, 0]],
    'A12': [Ahat[0, 1]],
    'A21': [Ahat[1, 0]],
    'A22': [Ahat[1, 1]],
    'lamFF11': 1,           # normalised to 1 by construction
    'lamFF22': 1,           # normalised to 1 by construction
    'lamBB11': Lam2[0],     # bank-level supply shock variance
    'lamBB22': Lam2[1],     # bank-level demand shock variance
    'se11':   [SEs[0]],
    'se12':   [SEs[2]],
    'se21':   [SEs[1]],
    'se22':   [SEs[3]],
    'se_R11': [SEs_R[0]],
    'se_R12': [SEs_R[2]],
    'se_R21': [SEs_R[1]],
    'se_R22': [SEs_R[3]],
    'Test_statistic':        [teststat],
    'Reject_null':           [rej],
    'Ratio_successful_optims': [np.sum(flag) / nstarts]
}

summary = pd.DataFrame(contents_df)

# Write header only on the first run; append data on subsequent runs.
header = not os.path.exists(output_summary_file)
summary.to_csv(output_summary_file, mode='a', header=header, index=False)
