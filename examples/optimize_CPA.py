import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import speed_of_light
from scipy.optimize import minimize

# Assurez-vous que le chemin vers la bibliothèque laserfun est correct
from laserfun.pulse import Pulse # Doit être la version que vous avez fournie

# --- 1. Définition des paramètres ---
center_wavelength_nm = 1550.0
fwhm_ps = 0.150  # 150 fs
energy_J = 100e-12  # 100 pJ

# Dispersion cible de l'étireur (à compenser)
GDD_stretcher_ps2 = 28.96
TOD_stretcher_ps3 = -1.063

# Paramètres fixes du compresseur
grating_lines_per_mm = 1200.5
m_order = -1
N_passes = 2


# --- 2. Création de l'impulsion initiale et étirée ---
# Utilisation de la définition de Pulse basée sur votre dernier fichier
pulse_initial = Pulse(
    pulse_type='sech',
    center_wavelength_nm=center_wavelength_nm,
    fwhm_ps=fwhm_ps,
    time_window_ps=2550.0, # Fenêtre large
    npts=2**20,
    epp=energy_J
)

pulse_stretched = pulse_initial.create_cloned_pulse()
pulse_stretched.chirp_pulse_W(GDD=GDD_stretcher_ps2, TOD=TOD_stretcher_ps3)


# --- 3. Optimisation Analytique (GDD seul) ---
print("--- Optimisation Partie 1 : GDD Seul (Analytique) ---")
pulse_gdd_optimized = pulse_stretched.create_cloned_pulse()

theta_i_deg_gdd_only = pulse_gdd_optimized.get_littrow_angle(grating_lines_per_mm, m_order=m_order)
print(f"Angle de Littrow (pour m={m_order}) : {theta_i_deg_gdd_only:.2f} deg")

GDD_target_s2 = -GDD_stretcher_ps2 * 1e-24
lambda_0_m = center_wavelength_nm * 1e-9
d_m = 1e-3 / grating_lines_per_mm
cos_theta_d_analytic = np.cos(np.deg2rad(theta_i_deg_gdd_only)) # En Littrow, theta_d = theta_i

L_g_m_analytic = -GDD_target_s2 * (2 * np.pi * speed_of_light**2 * d_m**2 * cos_theta_d_analytic**3) / (lambda_0_m**3 * m_order**2)
L_eff_m_gdd_only = L_g_m_analytic / N_passes
print(f"Distance calculée : {L_eff_m_gdd_only * 100:.2f} cm")

# On applique cette dispersion et on récupère les valeurs du compresseur
gdd_comp_analytic, tod_comp_analytic, fod_comp_analytic = pulse_gdd_optimized.apply_grating_compressor(
    grating_lines_per_mm, L_eff_m_gdd_only, theta_i_deg_gdd_only, m_order, N_passes)

print(f"\nDispersion du compresseur (analytique) :")
print(f"  GDD = {gdd_comp_analytic:.4f} ps^2")
print(f"  TOD = {tod_comp_analytic:.4f} ps^3")
print(f"  FOD = {fod_comp_analytic:.4f} ps^4")

print(f"Dispersion résiduelle (analytique) : GDD = {GDD_stretcher_ps2 + gdd_comp_analytic:.2e} ps^2, TOD = {TOD_stretcher_ps3 + tod_comp_analytic:.2e} ps^3")


# --- 4. Optimisation Numérique (GDD et TOD) ---
print("\n--- Optimisation Partie 2 : GDD et TOD (Numérique) ---")

target_gdd = -GDD_stretcher_ps2
target_tod = -TOD_stretcher_ps3

def cost_function(params, pulse_ref, target_gdd, target_tod):
    L_eff_m, theta_i_deg = params
    
    gdd, tod, _ = pulse_ref.calculate_grating_dispersion(
        grating_lines_per_mm, L_eff_m, theta_i_deg, m_order, N_passes)

    if np.isinf(gdd) or np.isinf(tod):
        return 1e12

    weight_gdd = 1.0
    weight_tod = 1.0

    err_gdd = ((gdd - target_gdd) / target_gdd)**2 if target_gdd != 0 else (gdd)**2
    err_tod = ((tod - target_tod) / target_tod)**2 if target_tod != 0 else (tod)**2
    
    return weight_gdd * err_gdd + weight_tod * err_tod

initial_guess = [L_eff_m_gdd_only, theta_i_deg_gdd_only]
bounds = [(0.001, 2.0), (-88, -50)]

optim_options = {'ftol': 1e-12, 'gtol': 1e-8}

result = minimize(
    cost_function,
    initial_guess,
    args=(pulse_initial, target_gdd, target_tod),
    method='L-BFGS-B',
    bounds=bounds,
    options=optim_options
)

L_eff_m_opt, theta_i_deg_opt = result.x
print("\nOptimisation terminée.")
print(f"  Succès : {result.success}")
print(f"  Message : {result.message}")
print(f"  Valeur finale de la fonction de coût : {result.fun:.2e}")
print(f"Angle optimal : {theta_i_deg_opt:.3f} deg")
print(f"Distance optimale : {L_eff_m_opt * 100:.3f} cm")

pulse_full_optimized = pulse_stretched.create_cloned_pulse()
gdd_comp_opt, tod_comp_opt, fod_comp_opt = pulse_full_optimized.apply_grating_compressor(
    grating_lines_per_mm, L_eff_m_opt, theta_i_deg_opt, m_order, N_passes)

print(f"\nDispersion du compresseur (optimisée) :")
print(f"  GDD = {gdd_comp_opt:.4f} ps^2")
print(f"  TOD = {tod_comp_opt:.4f} ps^3")
print(f"  FOD = {fod_comp_opt:.4f} ps^4")

print(f"Dispersion résiduelle (optimisée) : GDD = {GDD_stretcher_ps2 + gdd_comp_opt:.2e} ps^2, TOD = {TOD_stretcher_ps3 + tod_comp_opt:.2e} ps^3")


# --- 5. Visualisation des résultats ---
fig, ax = plt.subplots(figsize=(12, 7))

ax.plot(pulse_initial.t_ps, pulse_initial.it, 'k-', label=f'Initiale (FWHM = {pulse_initial.calc_width()*1e3:.1f} fs)')
ax.plot(pulse_gdd_optimized.t_ps, pulse_gdd_optimized.it, 'b--', label=f'Optimisée GDD seul (FWHM = {pulse_gdd_optimized.calc_width()*1e3:.1f} fs)')
ax.plot(pulse_full_optimized.t_ps, pulse_full_optimized.it, 'r-', lw=2, label=f'Optimisée GDD+TOD (FWHM = {pulse_full_optimized.calc_width()*1e3:.1f} fs)')

ax.set_title('Comparaison des optimisations de compresseur')
ax.set_xlabel('Temps (ps)')
ax.set_ylabel('Puissance (W)')
ax.set_xlim(-0.5, 0.5)
ax.legend()
ax.grid(True)
plt.tight_layout()
plt.show()