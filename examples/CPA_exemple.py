import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import speed_of_light
import copy

# Assurez-vous que le chemin vers la bibliothèque laserfun est correct
from laserfun.pulse import Pulse

# --- 1. Définition des paramètres ---
center_wavelength_nm = 1550.0
fwhm_ps = 0.150  # 150 fs
energy_J = 100e-12  # 100 pJ

GDD_stretcher_ps2 = 28.96
TOD_stretcher_ps3 = -1.063

grating_lines_per_mm = 1200.5
m_order = -1
N_passes = 2

# --- 2. Création de l'impulsion initiale ---
pulse_initial = Pulse(
    pulse_type='sech',
    center_wavelength_nm=center_wavelength_nm,
    fwhm_ps=fwhm_ps,
    time_window_ps=2550.0, # Fenêtre large comme vous l'aviez demandé
    npts=2**18,
    epp=energy_J
)

# --- 3. Étirement ---
pulse_stretched = pulse_initial.create_cloned_pulse()
pulse_stretched.chirp_pulse_W(GDD=GDD_stretcher_ps2, TOD=TOD_stretcher_ps3)


# --- 4. Compression ---
pulse_compressed = pulse_stretched.create_cloned_pulse()

theta_i_deg = pulse_compressed.get_littrow_angle(grating_lines_per_mm, m_order=m_order)
if np.isnan(theta_i_deg):
    raise ValueError("Impossible de calculer l'angle de Littrow avec ces paramètres.")

print(f"Angle de Littrow pour m=-1 calculé : {theta_i_deg:.2f} degrés")

GDD_target_s2 = -GDD_stretcher_ps2 * 1e-24
lambda_0_m = center_wavelength_nm * 1e-9
d_m = 1e-3 / grating_lines_per_mm
theta_i_rad = np.deg2rad(theta_i_deg)
cos_theta_d = np.cos(theta_i_rad) # En configuration de Littrow, theta_d = theta_i

c_mks = 299792458.0
L_g_m = -GDD_target_s2 * (2 * np.pi * c_mks**2 * d_m**2 * cos_theta_d**3) / (lambda_0_m**3 * m_order**2)
L_eff_m = L_g_m / N_passes
print(f"Distance effective L_eff requise : {L_eff_m * 100:.2f} cm")

gdd_comp, tod_comp, fod_comp = pulse_compressed.apply_grating_compressor(
    grating_lines_per_mm=grating_lines_per_mm,
    L_eff_m=L_eff_m,
    theta_i_deg=theta_i_deg,
    m_input=m_order,
    N_passes_formula=N_passes
)


# --- 5. Visualisation des résultats ---

# Calcul des FWHM pour les légendes
fwhm_initial_fs = pulse_initial.calc_width() * 1000
fwhm_stretched_ps = pulse_stretched.calc_width()
fwhm_compressed_fs = pulse_compressed.calc_width() * 1000

print(f"\nDurée impulsion initiale  : {fwhm_initial_fs:.1f} fs")
print(f"Durée impulsion étirée    : {fwhm_stretched_ps:.1f} ps")
print(f"Durée impulsion comprimée : {fwhm_compressed_fs:.1f} fs")

# Graphique 1 : Impulsion étirée
fig1, ax1 = plt.subplots(figsize=(10, 6))
ax1.plot(pulse_stretched.t_ps, pulse_stretched.it, 'b--', label=f'Étirée (FWHM = {fwhm_stretched_ps:.1f} ps)')
ax1.set_title('Impulsion après étirement')
ax1.set_xlabel('Temps (ps)')
ax1.set_ylabel('Puissance (W)')
ax1.grid(True, alpha=0.5)
ax1.legend()
# On laisse matplotlib choisir le zoom pour voir l'impulsion étirée en entier
ax1.set_xlim(-3*fwhm_stretched_ps, 3*fwhm_stretched_ps)


# Graphique 2 : Comparaison des impulsions initiale et comprimée (zoom)
fig2, ax2 = plt.subplots(figsize=(10, 6))
ax2.plot(pulse_initial.t_ps, pulse_initial.it, 'k-', label=f'Initiale (FWHM = {fwhm_initial_fs:.1f} fs)')
ax2.plot(pulse_compressed.t_ps, pulse_compressed.it, 'r-', lw=2, label=f'Comprimée (FWHM = {fwhm_compressed_fs:.1f} fs)')
ax2.set_title('Zoom sur les impulsions initiale et finale')
ax2.set_xlabel('Temps (ps)')
ax2.set_ylabel('Puissance (W)')
ax2.set_xlim(-5 * fwhm_ps, 5 * fwhm_ps) # Zoom sur quelques centaines de femtosecondes
ax2.grid(True, alpha=0.5)
ax2.legend()

plt.tight_layout()
plt.show()