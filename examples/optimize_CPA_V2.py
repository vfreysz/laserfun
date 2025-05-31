import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import speed_of_light
from scipy.optimize import minimize
import scipy.fftpack as fft # Utilisé pour l'autocorrélation

# Assurez-vous que le chemin vers la bibliothèque laserfun est correct
from laserfun.pulse import Pulse

# --- Fonctions utilitaires pour FWHM et Autocorrélation ---
def find_roots_for_fwhm(x, y_target_level_minus_half_max):
    """Trouve les racines où y_target_level_minus_half_max croise 0."""
    s = np.abs(np.diff(np.sign(y_target_level_minus_half_max))).astype(bool)
    if not np.any(s): return np.array([])
    return x[:-1][s] + np.diff(x)[s] * (0.0 - y_target_level_minus_half_max[:-1][s]) / \
                       (y_target_level_minus_half_max[1:][s] - y_target_level_minus_half_max[:-1][s])

def calc_fwhm_of_array(x_axis, y_array):
    """Calcule la FWHM d'une trace y_array donnée sur un x_axis."""
    max_y = np.max(y_array)
    if max_y == 0: return 0
    y_normalized = y_array / max_y
    roots = find_roots_for_fwhm(x_axis, y_normalized - 0.5)
    if len(roots) < 2: return 0
    # S'il y a plus de 2 racines (par ex. des oscillations),
    # on prend les plus éloignées pour la FWHM.
    return np.abs(roots[-1] - roots[0])

def calculate_intensity_autocorrelation(pulse_obj):
    intensity_t = pulse_obj.it
    spectrum_intensity_freq = fft.fft(intensity_t)
    power_spectrum_intensity = np.abs(spectrum_intensity_freq)**2
    autocorr_shifted = fft.ifft(power_spectrum_intensity)
    autocorr_centered = np.abs(fft.fftshift(autocorr_shifted))
    tau_axis = pulse_obj.t_ps
    return tau_axis, autocorr_centered

# --- 1. Définition des paramètres ---
center_wavelength_nm = 1550.0
fwhm_ps = 0.150  # 150 fs
energy_J = 100e-12  # 100 pJ

GDD_stretcher_ps2 = 28.96
TOD_stretcher_ps3 = -1.063
FOD_stretcher_ps4 = 0.0 # Initialement, pas de FOD de l'étireur pour ce test

grating_lines_per_mm = 1200.5
m_order = -1
N_passes = 2

# --- 2. Création de l'impulsion initiale et étirée ---
pulse_initial = Pulse(
    pulse_type='sech',
    center_wavelength_nm=center_wavelength_nm,
    fwhm_ps=fwhm_ps,
    time_window_ps=2550.0,
    npts=2**18,
    epp=energy_J
)

pulse_stretched = pulse_initial.create_cloned_pulse()
pulse_stretched.chirp_pulse_W(GDD=GDD_stretcher_ps2, TOD=TOD_stretcher_ps3, FOD=FOD_stretcher_ps4)

# --- 3. Optimisation Analytique (GDD seul) ---
print("--- Optimisation Partie 1 : GDD Seul (Analytique) ---")
pulse_gdd_optimized_stage = pulse_stretched.create_cloned_pulse() # Renommé pour clarté
theta_i_deg_gdd_only = pulse_gdd_optimized_stage.get_littrow_angle(grating_lines_per_mm, m_order=m_order)
GDD_target_s2 = -GDD_stretcher_ps2 * 1e-24
lambda_0_m = center_wavelength_nm * 1e-9
d_m = 1e-3 / grating_lines_per_mm
cos_theta_d_analytic = np.cos(np.deg2rad(theta_i_deg_gdd_only))
L_g_m_analytic = -GDD_target_s2 * (2 * np.pi * speed_of_light**2 * d_m**2 * cos_theta_d_analytic**3) / (lambda_0_m**3 * m_order**2)
L_eff_m_gdd_only = L_g_m_analytic / N_passes
print(f"Angle initial (Littrow m={m_order}) : {theta_i_deg_gdd_only:.2f} deg, Distance : {L_eff_m_gdd_only * 100:.2f} cm")
gdd_comp_an, tod_comp_an, fod_comp_an = pulse_gdd_optimized_stage.apply_grating_compressor(
    grating_lines_per_mm, L_eff_m_gdd_only, theta_i_deg_gdd_only, m_order, N_passes)
print(f"Dispersion résiduelle (analytique) : GDD={GDD_stretcher_ps2 + gdd_comp_an:.2e}, TOD={TOD_stretcher_ps3 + tod_comp_an:.2e}, FOD={FOD_stretcher_ps4 + fod_comp_an:.2e}")

# --- 4. Optimisation Numérique (GDD et TOD résiduels) ---
print("\n--- Optimisation Partie 2 : GDD et TOD Résiduels (Numérique) ---")
target_gdd_disp = -GDD_stretcher_ps2
target_tod_disp = -TOD_stretcher_ps3

def cost_function_disp_coeffs(params, pulse_ref, target_gdd, target_tod):
    L_eff_m, theta_i_deg = params
    gdd, tod, _ = pulse_ref.calculate_grating_dispersion(
        grating_lines_per_mm, L_eff_m, theta_i_deg, m_order, N_passes)
    if np.isinf(gdd): return 1e12
    err_gdd = ((gdd - target_gdd) / target_gdd)**2 if target_gdd != 0 else (gdd/0.01)**2
    err_tod = ((tod - target_tod) / target_tod)**2 if target_tod != 0 else (tod/0.001)**2
    return err_gdd + err_tod

initial_guess_disp = [L_eff_m_gdd_only, theta_i_deg_gdd_only]
bounds = [(0.001, 2.0), (-88, -50)] # (distance en m, angle en deg)
optim_options_disp = {'ftol': 1e-12, 'gtol': 1e-8}

result_disp = minimize(
    cost_function_disp_coeffs, initial_guess_disp, args=(pulse_initial, target_gdd_disp, target_tod_disp),
    method='L-BFGS-B', bounds=bounds, options=optim_options_disp)

L_eff_m_disp_opt, theta_i_deg_disp_opt = result_disp.x
print(f"Optimisation (GDD/TOD Coeffs) terminée. Coût: {result_disp.fun:.2e}")
print(f"  Angle : {theta_i_deg_disp_opt:.3f} deg, Distance : {L_eff_m_disp_opt * 100:.3f} cm")

pulse_disp_optimized_stage = pulse_stretched.create_cloned_pulse() # Renommé
gdd_c_disp, tod_c_disp, fod_c_disp = pulse_disp_optimized_stage.apply_grating_compressor(
    grating_lines_per_mm, L_eff_m_disp_opt, theta_i_deg_disp_opt, m_order, N_passes)
print(f"Dispersion résiduelle (opt. GDD/TOD Coeffs) : GDD={GDD_stretcher_ps2 + gdd_c_disp:.2e}, TOD={TOD_stretcher_ps3 + tod_c_disp:.2e}, FOD={FOD_stretcher_ps4 + fod_c_disp:.2e}")

# --- 5. Optimisation Numérique (FWHM de l'Autocorrélation) ---
print("\n--- Optimisation Partie 3 : FWHM Autocorrélation (Numérique) ---")

def cost_function_autocorr_fwhm(params, pulse_target_stretch):
    L_eff_m, theta_i_deg = params
    temp_pulse = pulse_target_stretch.create_cloned_pulse()
    # Important: apply_grating_compressor applique GDD, TOD ET FOD du compresseur
    gdd_comp, _, _ = temp_pulse.apply_grating_compressor(
         grating_lines_per_mm, L_eff_m, theta_i_deg, m_order, N_passes
    )
    if np.isinf(gdd_comp): return 1e12
    tau_axis, autocorr_trace = calculate_intensity_autocorrelation(temp_pulse)
    fwhm_autocorr = calc_fwhm_of_array(tau_axis, autocorr_trace)
    return fwhm_autocorr if fwhm_autocorr > 0 else 1e12

initial_guess_autocorr = [L_eff_m_disp_opt, theta_i_deg_disp_opt]
optim_options_autocorr = {'ftol': 1e-9, 'gtol': 1e-6} # Tolérances un peu moins strictes

result_autocorr = minimize(
    cost_function_autocorr_fwhm, initial_guess_autocorr, args=(pulse_stretched,),
    method='L-BFGS-B', bounds=bounds, options=optim_options_autocorr)

L_eff_m_autocorr_opt, theta_i_deg_autocorr_opt = result_autocorr.x
print(f"Optimisation (Autocorrélation FWHM) terminée. Coût (FWHM autocorr): {result_autocorr.fun:.4f} ps")
print(f"  Angle : {theta_i_deg_autocorr_opt:.3f} deg, Distance : {L_eff_m_autocorr_opt * 100:.3f} cm")

pulse_autocorr_optimized_stage = pulse_stretched.create_cloned_pulse() # Renommé
gdd_c_ac, tod_c_ac, fod_c_ac = pulse_autocorr_optimized_stage.apply_grating_compressor(
    grating_lines_per_mm, L_eff_m_autocorr_opt, theta_i_deg_autocorr_opt, m_order, N_passes)
print(f"Dispersion résiduelle (opt. Autocorr) : GDD={GDD_stretcher_ps2 + gdd_c_ac:.2e}, TOD={TOD_stretcher_ps3 + tod_c_ac:.2e}, FOD={FOD_stretcher_ps4 + fod_c_ac:.2e}")


# --- 6. NOUVELLE Optimisation Numérique (FWHM du Champ Réel) ---
print("\n--- Optimisation Partie 4 : FWHM Champ Réel (Numérique) ---")

def cost_function_direct_fwhm(params, pulse_target_stretch):
    L_eff_m, theta_i_deg = params
    temp_pulse = pulse_target_stretch.create_cloned_pulse()
    # On applique la dispersion complète du compresseur (GDD, TOD, FOD)
    gdd_comp, _, _ = temp_pulse.apply_grating_compressor(
         grating_lines_per_mm, L_eff_m, theta_i_deg, m_order, N_passes
    )
    if np.isinf(gdd_comp): return 1e12 # Si la géométrie est impossible

    # Calcule la FWHM de l'impulsion recompressée elle-même
    fwhm_pulse = temp_pulse.calc_width() # Utilise la méthode de la classe Pulse
                                         # qui calcule la FWHM de self.it
    return fwhm_pulse if fwhm_pulse > 0 else 1e12

# Point de départ pour cette optimisation : les résultats de l'optimisation par autocorrélation
initial_guess_direct_fwhm = [L_eff_m_autocorr_opt, theta_i_deg_autocorr_opt]

# On peut utiliser des tolérances strictes ici
optim_options_direct_fwhm = {'ftol': 1e-12, 'gtol': 1e-8}

result_direct_fwhm = minimize(
    cost_function_direct_fwhm,
    initial_guess_direct_fwhm,
    args=(pulse_stretched,),
    method='L-BFGS-B',
    bounds=bounds,
    options=optim_options_direct_fwhm
)

L_eff_m_final_opt, theta_i_deg_final_opt = result_direct_fwhm.x
print(f"Optimisation (FWHM Champ Réel) terminée. Coût (FWHM impulsion): {result_direct_fwhm.fun:.4f} ps")
print(f"  Angle final : {theta_i_deg_final_opt:.3f} deg, Distance finale : {L_eff_m_final_opt * 100:.3f} cm")

pulse_final_optimized = pulse_stretched.create_cloned_pulse()
gdd_c_final, tod_c_final, fod_c_final = pulse_final_optimized.apply_grating_compressor(
    grating_lines_per_mm, L_eff_m_final_opt, theta_i_deg_final_opt, m_order, N_passes)
print(f"Dispersion résiduelle (opt. FWHM Champ Réel) : GDD={GDD_stretcher_ps2 + gdd_c_final:.2e}, TOD={TOD_stretcher_ps3 + tod_c_final:.2e}, FOD={FOD_stretcher_ps4 + fod_c_final:.2e}")


# --- 7. Visualisation des résultats ---
fig, ax = plt.subplots(figsize=(14, 8)) # Légèrement plus large pour plus de traces

ax.plot(pulse_initial.t_ps, pulse_initial.it, 'k-', label=f'Initiale (FWHM = {pulse_initial.calc_width()*1e3:.1f} fs)')
ax.plot(pulse_gdd_optimized_stage.t_ps, pulse_gdd_optimized_stage.it, 'b--', alpha=0.7, label=f'Opt. GDD (FWHM = {pulse_gdd_optimized_stage.calc_width()*1e3:.1f} fs)')
ax.plot(pulse_disp_optimized_stage.t_ps, pulse_disp_optimized_stage.it, 'g-.', alpha=0.7, label=f'Opt. GDD+TOD Coeffs (FWHM = {pulse_disp_optimized_stage.calc_width()*1e3:.1f} fs)')
ax.plot(pulse_autocorr_optimized_stage.t_ps, pulse_autocorr_optimized_stage.it, 'm:', alpha=0.7, label=f'Opt. Autocorr (FWHM = {pulse_autocorr_optimized_stage.calc_width()*1e3:.1f} fs)')
ax.plot(pulse_final_optimized.t_ps, pulse_final_optimized.it, 'r-', lw=2, label=f'Opt. FWHM Champ Réel (FWHM = {pulse_final_optimized.calc_width()*1e3:.1f} fs)')

ax.set_title('Comparaison des optimisations de compresseur')
ax.set_xlabel('Temps (ps)')
ax.set_ylabel('Puissance (W)')
ax.set_xlim(-1.6, 1.6) # Zoom adapté
ax.legend(fontsize='small')
ax.grid(True)
plt.tight_layout()
plt.show()