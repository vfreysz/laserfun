import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import speed_of_light
from scipy.optimize import minimize
import scipy.fftpack as fft # Toujours nécessaire si des méthodes internes de Pulse l'utilisent

# Assurez-vous que le chemin vers la bibliothèque laserfun est correct
from laserfun.pulse import Pulse # Doit être la version que vous avez fournie

# --- Fonctions utilitaires pour FWHM (au cas où, bien que non optimisé directement) ---
def find_roots_for_fwhm(x, y_target_level_minus_half_max):
    s = np.abs(np.diff(np.sign(y_target_level_minus_half_max))).astype(bool)
    if not np.any(s): return np.array([])
    return x[:-1][s] + np.diff(x)[s] * (0.0 - y_target_level_minus_half_max[:-1][s]) / \
                       (y_target_level_minus_half_max[1:][s] - y_target_level_minus_half_max[:-1][s])

def calc_fwhm_of_array(x_axis, y_array):
    max_y = np.max(y_array)
    if max_y == 0: return 0
    y_normalized = y_array / max_y
    roots = find_roots_for_fwhm(x_axis, y_normalized - 0.5)
    if len(roots) < 2: return 0
    return np.abs(roots[-1] - roots[0])

# --- 1. Définition des paramètres ---
center_wavelength_nm = 1550.0
fwhm_ps = 0.150  # 150 fs
energy_J = 100e-12  # 100 pJ

GDD_stretcher_ps2 = 28.96
TOD_stretcher_ps3 = -1.063
FOD_stretcher_ps4 = 0.0 # FOD de l'étireur, gardée pour l'analyse de la dispersion résiduelle

grating_lines_per_mm = 1200.5
m_order = -1
N_passes = 2

# --- 2. Création de l'impulsion initiale et étirée ---
pulse_initial = Pulse(
    pulse_type='sech',
    center_wavelength_nm=center_wavelength_nm,
    fwhm_ps=fwhm_ps,
    time_window_ps=2550.0,
    npts=2**18, # Conformément à vos modifications précédentes
    epp=energy_J
)

pulse_stretched = pulse_initial.create_cloned_pulse()
pulse_stretched.chirp_pulse_W(GDD=GDD_stretcher_ps2, TOD=TOD_stretcher_ps3, FOD=FOD_stretcher_ps4)

# --- 3. Optimisation Analytique (GDD seul) - PARTIE 1 ---
print("--- Optimisation Partie 1 : GDD Seul (Analytique) ---")
pulse_gdd_optimized_stage = pulse_stretched.create_cloned_pulse()
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

# --- 4. Optimisation Numérique (GDD et TOD résiduels) - PARTIE 2 ---
print("\n--- Optimisation Partie 2 : GDD et TOD Résiduels (Numérique) ---")
target_gdd_disp = -GDD_stretcher_ps2
target_tod_disp = -TOD_stretcher_ps3

def cost_function_disp_coeffs(params, pulse_ref_for_disp_calc, target_gdd, target_tod):
    L_eff_m, theta_i_deg = params
    gdd, tod, _ = pulse_ref_for_disp_calc.calculate_grating_dispersion(
        grating_lines_per_mm, L_eff_m, theta_i_deg, m_order, N_passes)
    if np.isinf(gdd): return 1e12
    err_gdd = ((gdd - target_gdd) / target_gdd)**2 if target_gdd != 0 else (gdd/0.01)**2
    err_tod = ((tod - target_tod) / target_tod)**2 if target_tod != 0 else (tod/0.001)**2
    return err_gdd + err_tod

initial_guess_disp = [L_eff_m_gdd_only, theta_i_deg_gdd_only]
bounds = [(0.001, 2.0), (-88, -50)]
optim_options_disp = {'ftol': 1e-12, 'gtol': 1e-8}

result_disp = minimize(
    cost_function_disp_coeffs, initial_guess_disp, args=(pulse_initial, target_gdd_disp, target_tod_disp),
    method='L-BFGS-B', bounds=bounds, options=optim_options_disp)

L_eff_m_disp_opt, theta_i_deg_disp_opt = result_disp.x
print(f"Optimisation (GDD/TOD Coeffs) terminée. Coût: {result_disp.fun:.2e}")
print(f"  Angle : {theta_i_deg_disp_opt:.3f} deg, Distance : {L_eff_m_disp_opt * 100:.3f} cm")

pulse_disp_optimized_stage = pulse_stretched.create_cloned_pulse()
gdd_c_disp, tod_c_disp, fod_c_disp = pulse_disp_optimized_stage.apply_grating_compressor(
    grating_lines_per_mm, L_eff_m_disp_opt, theta_i_deg_disp_opt, m_order, N_passes)
print(f"Dispersion résiduelle (opt. GDD/TOD Coeffs) : GDD={GDD_stretcher_ps2 + gdd_c_disp:.2e}, TOD={TOD_stretcher_ps3 + tod_c_disp:.2e}, FOD={FOD_stretcher_ps4 + fod_c_disp:.2e}")

# --- 5. NOUVELLE Optimisation Numérique (Maximisation de la Puissance Crête) - PARTIE 3 ---
print("\n--- Optimisation Partie 3 : Maximisation Puissance Crête (Numérique) ---")

def cost_function_peak_power(params, pulse_target_stretch):
    L_eff_m, theta_i_deg = params
    
    temp_pulse = pulse_target_stretch.create_cloned_pulse()
    
    # Appliquer la dispersion complète du compresseur (GDD, TOD, FOD)
    # car nous voulons maximiser la puissance crête de l'impulsion "réelle"
    # que ce compresseur produirait.
    gdd_comp, tod_comp, fod_comp = temp_pulse.apply_grating_compressor(
         grating_lines_per_mm, L_eff_m, theta_i_deg, m_order, N_passes
    )
    
    if np.isinf(gdd_comp): # Si la géométrie est impossible
        return 1e12 # Retourner une très grande erreur (car on minimise -peak_power)

    peak_power = np.max(temp_pulse.it)
    
    return -peak_power # On minimise l'opposé de la puissance crête

# Point de départ pour cette optimisation : les résultats de l'optimisation GDD/TOD
initial_guess_peak_power = [L_eff_m_disp_opt, theta_i_deg_disp_opt]

# On peut utiliser des tolérances similaires ou légèrement moins strictes
# car la puissance crête peut être une fonction plus "plate" près de l'optimum.
optim_options_peak_power = {'ftol': 1e-9, 'gtol': 1e-6, 'eps':1e-9}

result_peak_power = minimize(
    cost_function_peak_power,
    initial_guess_peak_power,
    args=(pulse_stretched,),
    method='L-BFGS-B', # Ou 'Nelder-Mead' pourrait aussi être essayé
    bounds=bounds,
    options=optim_options_peak_power
)

L_eff_m_final_opt, theta_i_deg_final_opt = result_peak_power.x
print(f"Optimisation (Maximisation Puissance Crête) terminée.")
print(f"  Succès : {result_peak_power.success}, Message : {result_peak_power.message}")
print(f"  Valeur finale de la fonction de coût (-Puissance Crête) : {result_peak_power.fun:.3e} (W)") # C'est -Peak Power
print(f"  Puissance crête maximale atteinte : {-result_peak_power.fun:.3e} (W)")
print(f"Angle final : {theta_i_deg_final_opt:.3f} deg, Distance finale : {L_eff_m_final_opt * 100:.3f} cm")

pulse_peak_power_optimized = pulse_stretched.create_cloned_pulse()
gdd_c_final, tod_c_final, fod_c_final = pulse_peak_power_optimized.apply_grating_compressor(
    grating_lines_per_mm, L_eff_m_final_opt, theta_i_deg_final_opt, m_order, N_passes)

print(f"\nDispersion du compresseur (optimisée pour puissance crête) :")
print(f"  GDD = {gdd_c_final:.4f} ps^2")
print(f"  TOD = {tod_c_final:.4f} ps^3")
print(f"  FOD = {fod_c_final:.4f} ps^4")

print(f"Dispersion résiduelle (optimisée pour puissance crête) :")
print(f"  GDD = {GDD_stretcher_ps2 + gdd_c_final:.2e} ps^2")
print(f"  TOD = {TOD_stretcher_ps3 + tod_c_final:.2e} ps^3")
print(f"  FOD = {FOD_stretcher_ps4 + fod_c_final:.2e} ps^4")


# --- 6. Visualisation des résultats ---
fig, ax = plt.subplots(figsize=(12, 7))

ax.plot(pulse_initial.t_ps, pulse_initial.it, 'k-', label=f'Initiale (FWHM = {pulse_initial.calc_width()*1e3:.1f} fs)')
ax.plot(pulse_gdd_optimized_stage.t_ps, pulse_gdd_optimized_stage.it, 'c-.', alpha=0.7, label=f'Opt. GDD (FWHM = {pulse_gdd_optimized_stage.calc_width()*1e3:.1f} fs)')
ax.plot(pulse_disp_optimized_stage.t_ps, pulse_disp_optimized_stage.it, 'b--', alpha=0.7, label=f'Opt. GDD+TOD Coeffs (FWHM = {pulse_disp_optimized_stage.calc_width()*1e3:.1f} fs)')
ax.plot(pulse_peak_power_optimized.t_ps, pulse_peak_power_optimized.it, 'r-', lw=2, label=f'Opt. Puissance Crête (FWHM = {pulse_peak_power_optimized.calc_width()*1e3:.1f} fs)')

ax.set_title('Comparaison des optimisations de compresseur')
ax.set_xlabel('Temps (ps)')
ax.set_ylabel('Puissance (W)')
ax.set_xlim(-1.5, 1.5)
ax.legend(fontsize='small')
ax.grid(True)
plt.tight_layout()
plt.show()