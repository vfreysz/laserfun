import laserfun as lf
import numpy as np
import matplotlib.pyplot as plt

# --- Paramètres pour l'impulsion initiale (avant chirp) ---
center_wavelength_nm_initial = 1550
fwhm_fs_initial = 150
fwhm_ps_initial = fwhm_fs_initial * 1e-3

time_window_ps_initial = 10 # Fenêtre pour la première impulsion
npts_initial = 2**13        # Nombre de points pour la première impulsion
epp_J_initial = 1e-12

# --- Couleurs standard pour les graphiques ---
color_profile_temporal_std = 'darkslategrey'
color_phase_temporal_std = 'yellowgreen'

# --- Style Matplotlib ---
plt.style.use('seaborn-v0_8-whitegrid')

# --- GRAPHIQUE 1: Impulsion initiale ---
# Création de l'objet impulsion initiale
pulse_initial = lf.Pulse(
    pulse_type='gaussian',
    center_wavelength_nm=center_wavelength_nm_initial,
    fwhm_ps=fwhm_ps_initial,
    time_window_ps=time_window_ps_initial,
    npts=npts_initial,
    epp=epp_J_initial
)
# ... (Calculs et code de tracé pour l'impulsion initiale, identique au script précédent) ...
# Calcul des caractéristiques pour le tracé de l'impulsion initiale
time_initial_fs = pulse_initial.t_ps * 1000
temporal_intensity_initial = np.abs(pulse_initial.at)**2
normalized_temporal_intensity_initial = temporal_intensity_initial / np.max(temporal_intensity_initial)
temporal_phase_initial = np.unwrap(np.angle(pulse_initial.at))
fwhm_calculated_ps_initial = pulse_initial.calc_width(level=0.5)
fwhm_calculated_fs_initial = fwhm_calculated_ps_initial * 1000

wavelengths_nm_initial = pulse_initial.wavelength_nm
sort_indices_initial = np.argsort(wavelengths_nm_initial)
wavelengths_nm_sorted_initial = wavelengths_nm_initial[sort_indices_initial]
spectral_intensity_sorted_initial = (np.abs(pulse_initial.aw)**2)[sort_indices_initial]
normalized_spectral_intensity_initial = spectral_intensity_sorted_initial / np.max(spectral_intensity_sorted_initial)

fig_initial, (ax_temporal_initial, ax_spectrum_initial) = plt.subplots(1, 2, figsize=(14, 6))
ax_temporal_initial.plot(time_initial_fs, normalized_temporal_intensity_initial, color=color_profile_temporal_std, linewidth=2.5, label=f'Profil Temporel\nFWHM: {fwhm_calculated_fs_initial:.1f} fs')
ax_temporal_initial.fill_between(time_initial_fs, normalized_temporal_intensity_initial, color=color_profile_temporal_std, alpha=0.1)
ax_temporal_initial.set_xlabel('Temps (fs)', fontsize=12)
ax_temporal_initial.set_ylabel('Profil Temporel Normalisé (u.a.)', color=color_profile_temporal_std, fontsize=12)
ax_temporal_initial.tick_params(axis='y', labelcolor=color_profile_temporal_std)
ax_temporal_initial.set_ylim(0, 1.1)
xlim_temporal_initial_fs = 3 * fwhm_fs_initial
ax_temporal_initial.set_xlim(-xlim_temporal_initial_fs, xlim_temporal_initial_fs)
ax_temporal_initial.grid(True, linestyle=':', alpha=0.7)
ax_temporal_phase_initial = ax_temporal_initial.twinx()
ax_temporal_phase_initial.plot(time_initial_fs, temporal_phase_initial, color=color_phase_temporal_std, linestyle='--', linewidth=2, label='Phase Temporelle')
ax_temporal_phase_initial.set_ylabel('Phase Temporelle (rad)', color=color_phase_temporal_std, fontsize=12)
ax_temporal_phase_initial.tick_params(axis='y', labelcolor=color_phase_temporal_std)
min_phase_display_i = np.min(temporal_phase_initial[(time_initial_fs >= -xlim_temporal_initial_fs) & (time_initial_fs <= xlim_temporal_initial_fs)])
max_phase_display_i = np.max(temporal_phase_initial[(time_initial_fs >= -xlim_temporal_initial_fs) & (time_initial_fs <= xlim_temporal_initial_fs)])
padding_phase_i = (max_phase_display_i - min_phase_display_i) * 0.1 if (max_phase_display_i - min_phase_display_i) > 1e-6 else 1.0
ax_temporal_phase_initial.set_ylim(min_phase_display_i - padding_phase_i, max_phase_display_i + padding_phase_i)
ax_temporal_phase_initial.yaxis.grid(False)
ax_temporal_initial.set_title('Profil Temporel et Phase (Initiale)', fontsize=14)
lines_ti, labels_ti = ax_temporal_initial.get_legend_handles_labels()
lines_pi, labels_pi = ax_temporal_phase_initial.get_legend_handles_labels()
ax_temporal_phase_initial.legend(lines_ti + lines_pi, labels_ti + labels_pi, loc='upper right', fontsize=10)
ax_spectrum_initial.plot(wavelengths_nm_sorted_initial, normalized_spectral_intensity_initial, color=color_profile_temporal_std, linewidth=2)
ax_spectrum_initial.set_xlabel('Longueur d\'onde (nm)', fontsize=12)
ax_spectrum_initial.set_ylabel('Intensité Spectrale Normalisée (u.a.)', fontsize=12, color=color_profile_temporal_std)
ax_spectrum_initial.tick_params(axis='y', labelcolor=color_profile_temporal_std)
ax_spectrum_initial.set_title('Spectre de l\'Impulsion (Initiale)', fontsize=14)
ax_spectrum_initial.set_xlim(center_wavelength_nm_initial - 100, center_wavelength_nm_initial + 100)
ax_spectrum_initial.grid(True, linestyle=':', alpha=0.7)
fig_initial.tight_layout(pad=3.0)
plt.show()


# --- GRAPHIQUE 2: Impulsion après application de GDD et TOD ---
# Paramètres pour la simulation de l'impulsion chirpée
time_window_ps_chirped = 2500.0
npts_chirped = 2**18
gdd_ps2 = 28.96
tod_ps3 = -1.063

# Création de l'impulsion de base pour le chirp
pulse_base_for_chirp = lf.Pulse(
    pulse_type='gaussian',
    center_wavelength_nm=center_wavelength_nm_initial,
    fwhm_ps=fwhm_ps_initial,
    time_window_ps=time_window_ps_chirped,
    npts=npts_chirped,
    epp=epp_J_initial
)
pulse_chirped = pulse_base_for_chirp.create_cloned_pulse()
pulse_chirped.chirp_pulse_W(GDD=gdd_ps2, TOD=tod_ps3)
# ... (Calculs et code de tracé pour l'impulsion chirpée, identique au script précédent) ...
time_chirped_fs = pulse_chirped.t_ps * 1000
temporal_intensity_chirped = np.abs(pulse_chirped.at)**2
normalized_temporal_intensity_chirped = temporal_intensity_chirped / np.max(temporal_intensity_chirped)
temporal_phase_chirped = np.unwrap(np.angle(pulse_chirped.at))
fwhm_calculated_ps_chirped = pulse_chirped.calc_width(level=0.5)
fwhm_calculated_fs_chirped = fwhm_calculated_ps_chirped * 1000
wavelengths_nm_chirped = pulse_chirped.wavelength_nm
sort_indices_chirped = np.argsort(wavelengths_nm_chirped)
wavelengths_nm_sorted_chirped = wavelengths_nm_chirped[sort_indices_chirped]
spectral_intensity_sorted_chirped = (np.abs(pulse_chirped.aw)**2)[sort_indices_chirped]
normalized_spectral_intensity_chirped = spectral_intensity_sorted_chirped / np.max(spectral_intensity_sorted_chirped)
fig_chirped, (ax_temporal_chirped, ax_spectrum_chirped) = plt.subplots(1, 2, figsize=(14, 6))
ax_temporal_chirped.plot(time_chirped_fs, normalized_temporal_intensity_chirped, color=color_profile_temporal_std, linewidth=2.5, label=f'Profil Temporel (chirpé)\nFWHM: {fwhm_calculated_fs_chirped:.1f} fs')
ax_temporal_chirped.fill_between(time_chirped_fs, normalized_temporal_intensity_chirped, color=color_profile_temporal_std, alpha=0.1)
ax_temporal_chirped.set_xlabel('Temps (fs)', fontsize=12)
ax_temporal_chirped.set_ylabel('Profil Temporel Normalisé (u.a.)', color=color_profile_temporal_std, fontsize=12)
ax_temporal_chirped.tick_params(axis='y', labelcolor=color_profile_temporal_std)
ax_temporal_chirped.set_ylim(0, 1.1)
xlim_temporal_chirped_fs = (time_window_ps_chirped / 2.0) * 1000
ax_temporal_chirped.set_xlim(-xlim_temporal_chirped_fs, xlim_temporal_chirped_fs)
ax_temporal_chirped.grid(True, linestyle=':', alpha=0.7)
ax_temporal_phase_chirped = ax_temporal_chirped.twinx()
ax_temporal_phase_chirped.plot(time_chirped_fs, temporal_phase_chirped, color=color_phase_temporal_std, linestyle='--', linewidth=2, label='Phase Temporelle (chirpé)')
ax_temporal_phase_chirped.set_ylabel('Phase Temporelle (rad)', color=color_phase_temporal_std, fontsize=12)
ax_temporal_phase_chirped.tick_params(axis='y', labelcolor=color_phase_temporal_std)
relevant_phase_indices_c = (time_chirped_fs >= -xlim_temporal_chirped_fs) & (time_chirped_fs <= xlim_temporal_chirped_fs)
if np.any(relevant_phase_indices_c):
    min_phase_display_c = np.min(temporal_phase_chirped[relevant_phase_indices_c])
    max_phase_display_c = np.max(temporal_phase_chirped[relevant_phase_indices_c])
    padding_phase_c = (max_phase_display_c - min_phase_display_c) * 0.1 if (max_phase_display_c - min_phase_display_c) > 1e-6 else 1.0
    ax_temporal_phase_chirped.set_ylim(min_phase_display_c - padding_phase_c, max_phase_display_c + padding_phase_c)
else:
     ax_temporal_phase_chirped.set_ylim(-np.pi, np.pi)
ax_temporal_phase_chirped.yaxis.grid(False)
ax_temporal_chirped.set_title('Profil Temporel et Phase (Chirp GDD/TOD)', fontsize=14)
lines_tc, labels_tc = ax_temporal_chirped.get_legend_handles_labels()
lines_pc, labels_pc = ax_temporal_phase_chirped.get_legend_handles_labels()
ax_temporal_phase_chirped.legend(lines_tc + lines_pc, labels_tc + labels_pc, loc='upper right', fontsize=10)
ax_spectrum_chirped.plot(wavelengths_nm_sorted_chirped, normalized_spectral_intensity_chirped, color=color_profile_temporal_std, linewidth=2)
ax_spectrum_chirped.set_xlabel('Longueur d\'onde (nm)', fontsize=12)
ax_spectrum_chirped.set_ylabel('Intensité Spectrale Normalisée (u.a.)', fontsize=12, color=color_profile_temporal_std)
ax_spectrum_chirped.tick_params(axis='y', labelcolor=color_profile_temporal_std)
ax_spectrum_chirped.set_title('Spectre de l\'Impulsion (Chirp GDD/TOD)', fontsize=14)
ax_spectrum_chirped.set_xlim(center_wavelength_nm_initial - 100, center_wavelength_nm_initial + 100)
ax_spectrum_chirped.grid(True, linestyle=':', alpha=0.7)
fig_chirped.tight_layout(pad=3.0)
plt.show()


# --- GRAPHIQUE 3: Impulsion après compresseur à réseaux ---
# Paramètres du compresseur à réseaux
grating_lines_per_mm_comp = 1200.5  # traits/mm
incident_angle_deg_comp = 67.8#67.64    # degrés
L_eff_m_comp = 0.06773         # 69.7 mm converti en mètres
diffraction_order_comp = -1       # Ordre de diffraction
N_passes_formula_comp = 2         # Nombre de passages pour la formule

# L'impulsion d'entrée pour le compresseur est 'pulse_chirped'
pulse_to_compress = pulse_chirped.create_cloned_pulse()

# Application du compresseur à réseau
try:
    pulse_to_compress.apply_grating_compressor(
        grating_lines_per_mm=grating_lines_per_mm_comp,
        m=diffraction_order_comp,
        L_eff_m=L_eff_m_comp,
        theta_i_deg=incident_angle_deg_comp,
        N_passes_formula=N_passes_formula_comp
    ) #
except ValueError as e:
    print(f"Erreur lors de l'application du compresseur à réseau : {e}")
    # Si une erreur se produit, nous allons quand même essayer de tracer l'impulsion non compressée
    # pour éviter une interruption complète du script.
    # Ou alors, on pourrait choisir de ne pas afficher ce troisième graphique.
    # Pour l'instant, on continue avec pulse_to_compress (qui serait identique à pulse_chirped).

# Calcul des caractéristiques de l'impulsion COMPRESSÉE
time_compressed_fs = pulse_to_compress.t_ps * 1000
temporal_intensity_compressed = np.abs(pulse_to_compress.at)**2
normalized_temporal_intensity_compressed = temporal_intensity_compressed / np.max(temporal_intensity_compressed)
temporal_phase_compressed = np.unwrap(np.angle(pulse_to_compress.at))
fwhm_calculated_ps_compressed = pulse_to_compress.calc_width(level=0.5)
fwhm_calculated_fs_compressed = fwhm_calculated_ps_compressed * 1000

wavelengths_nm_compressed = pulse_to_compress.wavelength_nm
sort_indices_compressed = np.argsort(wavelengths_nm_compressed)
wavelengths_nm_sorted_compressed = wavelengths_nm_compressed[sort_indices_compressed]
spectral_intensity_sorted_compressed = (np.abs(pulse_to_compress.aw)**2)[sort_indices_compressed]
normalized_spectral_intensity_compressed = spectral_intensity_sorted_compressed / np.max(spectral_intensity_sorted_compressed)

# Création de la troisième figure
fig_compressed, (ax_temporal_compressed, ax_spectrum_compressed) = plt.subplots(1, 2, figsize=(14, 6))

# Tracé temporel de l'impulsion COMPRESSÉE (gauche)
ax_temporal_compressed.plot(time_compressed_fs, normalized_temporal_intensity_compressed, color=color_profile_temporal_std, linewidth=2.5, label=f'Profil Temporel (compressé)\nFWHM: {fwhm_calculated_fs_compressed:.1f} fs')
ax_temporal_compressed.fill_between(time_compressed_fs, normalized_temporal_intensity_compressed, color=color_profile_temporal_std, alpha=0.1)
ax_temporal_compressed.set_xlabel('Temps (fs)', fontsize=12)
ax_temporal_compressed.set_ylabel('Profil Temporel Normalisé (u.a.)', color=color_profile_temporal_std, fontsize=12)
ax_temporal_compressed.tick_params(axis='y', labelcolor=color_profile_temporal_std)
ax_temporal_compressed.set_ylim(0, 1.1)
# Ajuster les limites pour voir l'impulsion compressée, par exemple +/- 500 fs ou basé sur la nouvelle FWHM
xlim_temporal_compressed_fs = max(5 * fwhm_calculated_fs_compressed, 300) # Au moins +/- 300fs
ax_temporal_compressed.set_xlim(-xlim_temporal_compressed_fs, xlim_temporal_compressed_fs)
ax_temporal_compressed.grid(True, linestyle=':', alpha=0.7)

ax_temporal_phase_compressed = ax_temporal_compressed.twinx()
ax_temporal_phase_compressed.plot(time_compressed_fs, temporal_phase_compressed, color=color_phase_temporal_std, linestyle='--', linewidth=2, label='Phase Temporelle (compressé)')
ax_temporal_phase_compressed.set_ylabel('Phase Temporelle (rad)', color=color_phase_temporal_std, fontsize=12)
ax_temporal_phase_compressed.tick_params(axis='y', labelcolor=color_phase_temporal_std)
relevant_phase_indices_comp = (time_compressed_fs >= -xlim_temporal_compressed_fs) & (time_compressed_fs <= xlim_temporal_compressed_fs)
if np.any(relevant_phase_indices_comp):
    min_phase_display_comp = np.min(temporal_phase_compressed[relevant_phase_indices_comp])
    max_phase_display_comp = np.max(temporal_phase_compressed[relevant_phase_indices_comp])
    padding_phase_comp = (max_phase_display_comp - min_phase_display_comp) * 0.1 if (max_phase_display_comp - min_phase_display_comp) > 1e-6 else 1.0
    ax_temporal_phase_compressed.set_ylim(min_phase_display_comp - padding_phase_comp, max_phase_display_comp + padding_phase_comp)
else:
    ax_temporal_phase_compressed.set_ylim(-np.pi,np.pi) # Fallback

ax_temporal_phase_compressed.yaxis.grid(False)
ax_temporal_compressed.set_title('Profil Temporel et Phase (Après Compression)', fontsize=14)

lines_tcomp, labels_tcomp = ax_temporal_compressed.get_legend_handles_labels()
lines_pcomp, labels_pcomp = ax_temporal_phase_compressed.get_legend_handles_labels()
ax_temporal_phase_compressed.legend(lines_tcomp + lines_pcomp, labels_tcomp + labels_pcomp, loc='upper right', fontsize=10)

# Tracé du spectre de l'impulsion COMPRESSÉE (droite)
ax_spectrum_compressed.plot(wavelengths_nm_sorted_compressed, normalized_spectral_intensity_compressed, color=color_profile_temporal_std, linewidth=2)
ax_spectrum_compressed.set_xlabel('Longueur d\'onde (nm)', fontsize=12)
ax_spectrum_compressed.set_ylabel('Intensité Spectrale Normalisée (u.a.)', fontsize=12, color=color_profile_temporal_std)
ax_spectrum_compressed.tick_params(axis='y', labelcolor=color_profile_temporal_std)
ax_spectrum_compressed.set_title('Spectre de l\'Impulsion (Après Compression)', fontsize=14)
ax_spectrum_compressed.set_xlim(center_wavelength_nm_initial - 100, center_wavelength_nm_initial + 100)
ax_spectrum_compressed.grid(True, linestyle=':', alpha=0.7)

fig_compressed.tight_layout(pad=3.0)
plt.show()