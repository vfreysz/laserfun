import matplotlib.pyplot as plt
from laserfun.pulse import Pulse, c_nmps # Importer c_nmps depuis laserfun.pulse
from laserfun.fiber import Fiber # Importer la classe Fiber
from laserfun.nlse import NLSE # Importer la fonction NLSE
import numpy as np

# --- 1. Définition des paramètres de l'impulsion initiale ---
fwhm_fs = 250  # Durée de l'impulsion en femtosecondes
fwhm_ps = fwhm_fs * 1e-3  # Conversion en picosecondes pour laserfun

# === MODIFICATION : Énergie par impulsion réduite ===
energie_pJ = 500 # Nouvelle énergie en picoJoules
energie_J = energie_pJ * 1e-12  # Conversion en Joules
# =================================================

longueur_onde_centrale_nm = 1555  # Longueur d'onde centrale en nm
GDD_initial_ps2 = 0.000 # GDD initiale (ps^2)

time_window_ps = 100 * fwhm_ps 
npts_pulse = 2**14 

# --- 2. Création de l'objet Pulse initial ---
impulsion_initiale = Pulse(
    pulse_type='sech',
    fwhm_ps=fwhm_ps,
    epp=energie_J,
    center_wavelength_nm=longueur_onde_centrale_nm,
    time_window_ps=time_window_ps,
    npts=npts_pulse,
    GDD=GDD_initial_ps2
)

print(f"Impulsion initiale créée:")
print(f"  - FWHM (calculé) : {impulsion_initiale.calc_width()*1e3:.2f} fs")
print(f"  - Énergie par impulsion : {impulsion_initiale.epp*1e12:.2f} pJ") # Affichage en pJ
print(f"  - Longueur d'onde centrale : {impulsion_initiale.center_wavelength_nm:.2f} nm")
print(f"  - GDD initiale : {GDD_initial_ps2:.4f} ps^2")

# --- Affichage de l'impulsion initiale ---
fig_temp_init, ax_temp_init = impulsion_initiale.plot_temporal(show_phase=True)
ax_temp_init.set_title(f"Initiale : Profil Temporel ({fwhm_fs} fs, {energie_pJ} pJ)")
ax_temp_init.set_xlim(-10 * fwhm_ps, 10 * fwhm_ps)
if len(fig_temp_init.get_axes()) > 1:
    ax_phase_temp_init = fig_temp_init.get_axes()[1]
    ax_phase_temp_init.set_ylim(-2 * np.pi, 2 * np.pi)
fig_temp_init.tight_layout()

fig_spec_init, ax_spec_init = impulsion_initiale.plot_spectral(x_axis='wavelength', show_phase=True)
ax_spec_init.set_title(f"Initiale : Spectre ({longueur_onde_centrale_nm} nm)")
delta_nu_THz_init = 0.315 / fwhm_ps
delta_lambda_fwhm_nm_init = (longueur_onde_centrale_nm**2 / c_nmps) * delta_nu_THz_init
ax_spec_init.set_xlim(longueur_onde_centrale_nm - 10 * delta_lambda_fwhm_nm_init, # Vue plus large pour la phase
                      longueur_onde_centrale_nm + 10 * delta_lambda_fwhm_nm_init)
if len(fig_spec_init.get_axes()) > 1:
    ax_phase_spec_init = fig_spec_init.get_axes()[1]
    ax_phase_spec_init.set_ylim(-2 * np.pi, 2 * np.pi)
fig_spec_init.tight_layout()


# --- 3. Définition des paramètres de la fibre PM1550 ---
longueur_fibre_m = 10.0
D_ps_nm_km = 18.0
S_D_ps_nm2_km = 0.06

beta2_fibre_ps2_m = -(D_ps_nm_km * 1e-3) * (longueur_onde_centrale_nm**2) / (2 * np.pi * c_nmps)
term1_factor_beta3 = (longueur_onde_centrale_nm / (2 * np.pi * c_nmps))**2
beta3_fibre_ps3_m = term1_factor_beta3 * ( (S_D_ps_nm2_km * 1e-3 * longueur_onde_centrale_nm**2) + \
                                      (2 * D_ps_nm_km * 1e-3 * longueur_onde_centrale_nm) )

gamma_W_km = 1.5
gamma_fibre_W_m = gamma_W_km * 1e-3
pertes_dB_km = 0.2
pertes_fibre_dB_m = pertes_dB_km * 1e-3

fibre_pm1550 = Fiber(
    length=longueur_fibre_m,
    center_wl_nm=longueur_onde_centrale_nm,
    dispersion_format='GVD',
    dispersion=[beta2_fibre_ps2_m, beta3_fibre_ps3_m],
    gamma_W_m=gamma_fibre_W_m,
    loss_dB_per_m=pertes_fibre_dB_m
)
print(f"\nFibre PM1550 créée:")
print(f"  - Longueur : {fibre_pm1550.length:.1f} m")
print(f"  - beta2 : {beta2_fibre_ps2_m:.2e} ps^2/m")
print(f"  - beta3 : {beta3_fibre_ps3_m:.2e} ps^3/m")
print(f"  - gamma : {fibre_pm1550.gamma:.2e} W^-1 m^-1")
print(f"  - pertes : {fibre_pm1550.alpha:.2e} dB/m")

# --- 4. Propagation de l'impulsion dans la fibre ---
print(f"\nDébut de la propagation NLSE sur {longueur_fibre_m} m...")
resultats_propagation = NLSE(
    pulse=impulsion_initiale,
    fiber=fibre_pm1550,
    nsaves=200, 
    raman=True, 
    shock=True, 
    print_status=False, # Mettre à True pour suivre la progression
    atol=1e-5, # Tolérance absolue un peu plus stricte
    rtol=1e-5  # Tolérance relative un peu plus stricte
)
impulsion_sortie = resultats_propagation.pulse_out
print("Propagation terminée.")

print(f"\nImpulsion en sortie de fibre:")
print(f"  - FWHM (calculé) : {impulsion_sortie.calc_width()*1e3:.2f} fs")
print(f"  - Énergie par impulsion : {impulsion_sortie.epp*1e12:.2f} pJ (après pertes)")


# --- 5. Affichage du profil temporel après propagation ---
fig_temp_sortie, ax_temp_sortie = impulsion_sortie.plot_temporal(show_phase=True)
ax_temp_sortie.set_title(f"Après {longueur_fibre_m}m PM1550 : Profil Temporel")
fwhm_sortie_ps = impulsion_sortie.calc_width()
# Ajuster le xlim pour voir l'impulsion qui peut s'être élargie ou compressée
ax_temp_sortie.set_xlim(impulsion_sortie.t_ps[0]/2, impulsion_sortie.t_ps[-1]/2) # Visualiser une bonne partie


if len(fig_temp_sortie.get_axes()) > 1:
    ax_phase_temp_sortie = fig_temp_sortie.get_axes()[1]
    # La variation de phase peut être plus importante après propagation non-linéaire
    ax_phase_temp_sortie.set_ylim(-10 * np.pi, 10 * np.pi) 
fig_temp_sortie.tight_layout()

# --- 6. Affichage du spectre en longueur d'onde après propagation ---
fig_spec_sortie, ax_spec_sortie = impulsion_sortie.plot_spectral(x_axis='wavelength', show_phase=True)
ax_spec_sortie.set_title(f"Après {longueur_fibre_m}m PM1550 : Spectre")

# La largeur spectrale peut avoir beaucoup changé.
# On peut centrer sur la wl initiale et choisir une plage large, ou adapter.
ax_spec_sortie.set_xlim(longueur_onde_centrale_nm - 50, longueur_onde_centrale_nm + 50) # Plage plus large pour SPM

if len(fig_spec_sortie.get_axes()) > 1:
    ax_phase_spec_sortie = fig_spec_sortie.get_axes()[1]
    ax_phase_spec_sortie.set_ylim(-20 * np.pi, 20 * np.pi) # Potentiellement grande variation de phase spectrale
fig_spec_sortie.tight_layout()

# --- 7. Afficher tous les graphiques ---
plt.show()