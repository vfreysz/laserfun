import laserfun as lf
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import speed_of_light # c en m/s

# --- Fonctions Utilitaires (copiées ici pour autonomie du script) ---
def get_littrow_angle_rad(longueur_onde_m: float, periode_reseau_m: float) -> float:
    """
    Calcule l'angle de Littrow en radians.
    """
    if not (2 * periode_reseau_m > 0):
        raise ValueError("La période du réseau doit être positive.")
    val_arcsin = longueur_onde_m / (2 * periode_reseau_m)
    if np.abs(val_arcsin) > 1.000001: # Petite tolérance pour erreurs de float
        raise ValueError(f"Impossible de calculer l'angle de Littrow, |lambda / (2d)| = {np.abs(val_arcsin):.3f} > 1.")
    # Clipper la valeur si elle est très légèrement hors de [-1, 1] à cause d'erreurs de float
    val_arcsin = np.clip(val_arcsin, -1.0, 1.0)
    return np.arcsin(val_arcsin)

def plot_pulse_characteristics(pulse_obj, title_prefix=""):
    """
    Affiche le profil temporel et le spectre d'une impulsion.
    """
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    
    current_fwhm_ps = pulse_obj.calc_width(level=0.5)
    
    axs[0].plot(pulse_obj.t_ps, np.abs(pulse_obj.at)**2, label=f'FWHM: {current_fwhm_ps*1e3:.2f} fs')
    axs[0].set_xlabel('Temps (ps)')
    axs[0].set_ylabel('Intensité (u.a.)')
    axs[0].set_title(f'{title_prefix} - Profil Temporel')
    axs[0].set_xlim(-2*max(fwhm_ps_initial, current_fwhm_ps, 0.1), 2*max(fwhm_ps_initial, current_fwhm_ps, 0.1))
    axs[0].legend()
    axs[0].grid(True, alpha=0.5)
    
    sort_indices = np.argsort(pulse_obj.wavelength_nm)
    wavelength_nm_sorted = pulse_obj.wavelength_nm[sort_indices]
    aw_sorted = pulse_obj.aw[sort_indices]
    
    axs[1].plot(wavelength_nm_sorted, np.abs(aw_sorted)**2)
    axs[1].set_xlabel('Longueur d\'onde (nm)')
    axs[1].set_ylabel('Intensité spectrale (u.a.)')
    axs[1].set_title(f'{title_prefix} - Spectre')
    axs[1].set_xlim(center_wavelength_nm - 50, center_wavelength_nm + 50)
    axs[1].grid(True, alpha=0.5)
    
    plt.tight_layout()
    plt.show()

# --- Paramètres ---
# Impulsion initiale
center_wavelength_nm = 1555.0  # nm
fwhm_fs_initial = 150.0          # fs
fwhm_ps_initial = fwhm_fs_initial * 1e-3 # Convertir fs en ps

# Paramètres du CFBG (depuis l'image précédente)
gdd_cfbg_ps2 = 28.96   # ps^2
tod_cfbg_ps3 = -1.063  # ps^3

# Paramètres de simulation pour l'impulsion
time_window_ps = 2500.0 
npts = 2**18         
epp_J = 1e-9          

# Paramètres pour le compresseur à réseau (d'après image image_ef908b.png)
grating_lines_per_mm_comp = 1200.5  # traits/mm
incident_angle_deg_comp = 68.5    # degrés
diffraction_order_comp = -1       # Nécessaire pour que les formules GDD/TOD fonctionnent
N_passes_formula_comp = 2         # Nombre de "passes" pour la formule GDD/TOD
# Longueur effective calculée pour cibler l'inversion du GDD du CFBG
effective_path_length_m_comp = 0.06500 # mètres (6.50 cm)


# --- Calcul de l'angle de Littrow ---
print("--- Calcul de l'angle de Littrow pour le réseau du compresseur ---")
lambda_0_m_pulse = center_wavelength_nm * 1e-9
d_m_grating = (1e-3 / grating_lines_per_mm_comp)
try:
    littrow_angle_rad = get_littrow_angle_rad(lambda_0_m_pulse, d_m_grating)
    littrow_angle_deg = np.rad2deg(littrow_angle_rad)
    print(f"Pour une longueur d'onde de {center_wavelength_nm} nm et {grating_lines_per_mm_comp} L/mm :")
    print(f"  Période du réseau (d) : {d_m_grating*1e9:.2f} nm")
    print(f"  Angle de Littrow (θ_L) : {littrow_angle_deg:.2f} degrés")
    print(f"  Angle d'incidence utilisé : {incident_angle_deg_comp:.2f} degrés (proche de Littrow)\n")
except ValueError as e:
    print(f"Erreur lors du calcul de l'angle de Littrow : {e}\n")


# --- Étape 1: Création de l'impulsion initiale ---
print(f"Création de l'impulsion initiale à {center_wavelength_nm} nm, FWHM: {fwhm_fs_initial} fs...")
pulse_initiale = lf.Pulse(
    pulse_type='gaussian', 
    center_wavelength_nm=center_wavelength_nm,
    fwhm_ps=fwhm_ps_initial,
    time_window_ps=time_window_ps,
    npts=npts,
    epp=epp_J
)
plot_pulse_characteristics(pulse_initiale, "Impulsion Initiale")
print(f"Durée FWHM initiale calculée : {pulse_initiale.calc_width()*1e3:.2f} fs\n")

# --- Étape 2: Application du CFBG ---
print(f"Application du CFBG : GDD = {gdd_cfbg_ps2} ps^2, TOD = {tod_cfbg_ps3} ps^3...")
pulse_apres_cfbg = pulse_initiale.create_cloned_pulse() 
pulse_apres_cfbg.chirp_pulse_W(GDD=gdd_cfbg_ps2, TOD=tod_cfbg_ps3)
plot_pulse_characteristics(pulse_apres_cfbg, "Impulsion après CFBG")
fwhm_apres_cfbg_fs = pulse_apres_cfbg.calc_width()*1e3
print(f"Durée FWHM après CFBG calculée : {fwhm_apres_cfbg_fs:.2f} fs\n")

# --- Étape 3: Compression de l'impulsion avec apply_grating_compressor ---
print("Compression de l'impulsion avec le compresseur à réseau...")
pulse_compressee_reseau = pulse_apres_cfbg.create_cloned_pulse()

print(f"  Paramètres du compresseur à réseau utilisés :")
print(f"    Traits/mm : {grating_lines_per_mm_comp}")
print(f"    Ordre m (formule) : {diffraction_order_comp}")
print(f"    Angle d'incidence : {incident_angle_deg_comp}°")
print(f"    L_eff : {effective_path_length_m_comp:.4f} m")
print(f"    N_passes (formule) : {N_passes_formula_comp}")

try:
    # La méthode apply_grating_compressor dans Pulse devrait afficher les GDD/TOD calculés
    pulse_compressee_reseau.apply_grating_compressor(
        grating_lines_per_mm=grating_lines_per_mm_comp,
        m=diffraction_order_comp,
        L_eff_m=effective_path_length_m_comp,
        theta_i_deg=incident_angle_deg_comp,
        N_passes_formula=N_passes_formula_comp
    )
    # Le message sur les GDD/TOD appliqués sera affiché par la méthode elle-même.
    plot_pulse_characteristics(pulse_compressee_reseau, "Impulsion Compressée par Réseau")
    fwhm_compressee_fs = pulse_compressee_reseau.calc_width()*1e3
    print(f"Durée FWHM après compression par réseau calculée : {fwhm_compressee_fs:.2f} fs\n")

    if fwhm_compressee_fs < fwhm_apres_cfbg_fs:
        print("L'impulsion a été compressée par le système de réseaux.")
        # Vérifier si la compression est proche de la durée initiale
        # Nous nous attendons à une bonne compression car L_eff_m a été calculé pour cela.
        if np.isclose(fwhm_compressee_fs, fwhm_fs_initial, rtol=0.05): # Tolérance de 5%
            print("La compression a bien ramené l'impulsion près de sa durée initiale !")
        elif fwhm_compressee_fs < fwhm_fs_initial * 1.1 : # Un peu plus large mais quand même bien compressé
             print(f"Bonne compression obtenue, mais légèrement différente de l'impulsion initiale (cible: {fwhm_fs_initial:.2f} fs).")
        else:
            print("La durée de l'impulsion compressée est significativement différente de la durée initiale, vérifier les TOD.")
    else:
        print("L'impulsion ne semble pas avoir été compressée, voire a été davantage étirée.")
        print("  -> Vérifiez les paramètres du compresseur à réseau et les GDD/TOD calculés.")

except ValueError as e:
    print(f"Erreur lors de l'application du compresseur à réseau : {e}")
    print("  -> Vérifiez les paramètres du réseau (angle, densité de traits) pour éviter des conditions non physiques.")
except AttributeError:
    print("\nERREUR: La méthode 'apply_grating_compressor' ne semble pas être définie dans la classe Pulse.")
    print("Veuillez vous assurer que vous avez ajouté cette méthode à votre fichier laserfun/pulse.py.\n")