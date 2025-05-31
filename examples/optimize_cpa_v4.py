import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import speed_of_light
from scipy.optimize import minimize
import scipy.fftpack as fft
import copy # Pour deepcopy si create_cloned_pulse ne suffit pas

# Assurez-vous que le chemin vers la bibliothèque laserfun est correct
# et que vous utilisez le fichier pulse.py que vous avez fourni.
from laserfun.pulse import Pulse

# --- Fonctions utilitaires ---

def extract_pulse_dispersion(pulse_obj, fit_order=9):
    """
    Extrait la GDD, TOD, et FOD d'un objet Pulse en ajustant sa phase spectrale.
    """
    if not hasattr(pulse_obj, 'aw') or not hasattr(pulse_obj, 'w_THz'):
        raise ValueError("L'objet Pulse doit avoir 'aw' et 'w_THz'.")

    omega_centered_rad_ps = pulse_obj.v_THz # Pulsations relatives (déjà centrées et fftshiftées)
    
    # Phase spectrale déroulée du spectre fftshifté
    spectral_phase_rad = np.unwrap(np.angle(fft.fftshift(pulse_obj.aw)))
    
    if len(omega_centered_rad_ps) != len(spectral_phase_rad):
        # Correction pour les cas où Npts est impair et fftshift/fftfreq causent un décalage de 1
        min_len = min(len(omega_centered_rad_ps), len(spectral_phase_rad))
        omega_centered_rad_ps = omega_centered_rad_ps[:min_len]
        spectral_phase_rad = spectral_phase_rad[:min_len]
        if verbose_main:
             print(f"Warning: Ajustement de la taille des tableaux pour polyfit dans extract_pulse_dispersion de {len(pulse_obj.v_THz)} à {min_len}")


    poly_coeffs = np.polyfit(omega_centered_rad_ps, spectral_phase_rad, fit_order)
    
    gdd_ps2, tod_ps3, fod_ps4 = 0.0, 0.0, 0.0
    if fit_order >= 2:
        gdd_ps2 = 2 * poly_coeffs[fit_order - 2]
    if fit_order >= 3:
        tod_ps3 = 6 * poly_coeffs[fit_order - 3]
    if fit_order >= 4:
        fod_ps4 = 24 * poly_coeffs[fit_order - 4]
    return gdd_ps2, tod_ps3, fod_ps4

def find_roots_for_fwhm(x, y_target_level_minus_ref):
    s = np.abs(np.diff(np.sign(y_target_level_minus_ref))).astype(bool)
    if not np.any(s): return np.array([])
    return x[:-1][s] + np.diff(x)[s] * (0.0 - y_target_level_minus_ref[:-1][s]) / \
                       (y_target_level_minus_ref[1:][s] - y_target_level_minus_ref[:-1][s])

def calc_fwhm_of_array(x_axis, y_array):
    max_y = np.max(y_array)
    if max_y == 0: return 0.0
    y_normalized = y_array / max_y
    roots = find_roots_for_fwhm(x_axis, y_normalized - 0.5)
    if len(roots) < 2: return 0.0
    return np.abs(roots[-1] - roots[0])

# ==============================================================================
# === FONCTION D'OPTIMISATION DU COMPRESSEUR ===================================
# ==============================================================================

def optimize_compressor(
    pulse_stretched_input, # Renommé pour éviter confusion avec variable globale
    grating_lines_per_mm,
    m_order_comp=-1, # Renommé pour clarté
    N_passes_comp=2, # Renommé pour clarté
    optimization_level='peak_power',
    verbose=True
):
    if not isinstance(pulse_stretched_input, Pulse):
        raise TypeError("pulse_stretched_input doit être une instance de la classe Pulse.")

    center_wavelength_nm_opt = pulse_stretched_input.center_wavelength_nm
    
    if verbose: print("--- Analyse de la dispersion de l'impulsion étirée ---")
    gdd_in_ps2, tod_in_ps3, _ = extract_pulse_dispersion(pulse_stretched_input, fit_order=3) # FOD non nécessaire pour cible
    if verbose:
        print(f"Dispersion de 'pulse_stretched_input': GDD={gdd_in_ps2:.4f} ps^2, TOD={tod_in_ps3:.4f} ps^3")

    target_gdd_comp = -gdd_in_ps2
    target_tod_comp = -tod_in_ps3

    # --- Partie 1 : Calcul Analytique (GDD seul) ---
    if verbose: print("\n--- Optimisation Partie 1 : GDD Seul (Analytique) ---")
    temp_pulse_for_littrow = pulse_stretched_input.create_cloned_pulse()
    theta_i_deg_gdd_only = temp_pulse_for_littrow.get_littrow_angle(grating_lines_per_mm, m_order=m_order_comp)
    if np.isnan(theta_i_deg_gdd_only):
        print("Warning: Calcul de l'angle de Littrow initial a échoué, tentative avec un angle par défaut.")
        theta_i_deg_gdd_only = -60.0 # Valeur par défaut si Littrow échoue
        
    GDD_target_s2_comp = target_gdd_comp * 1e-24
    lambda_0_m_opt = center_wavelength_nm_opt * 1e-9
    d_m_opt = 1e-3 / grating_lines_per_mm
    
    # Vérification pour cos_theta_d_analytic
    cos_theta_d_analytic_val = np.cos(np.deg2rad(theta_i_deg_gdd_only))
    if np.isclose(cos_theta_d_analytic_val,0):
        print("Warning: cos(theta_d) proche de zéro pour calcul analytique de L_eff. Utilisation d'une valeur L_eff par défaut.")
        L_eff_m_gdd_only = 0.1 # 10 cm par défaut
    else:
        L_g_m_analytic = -GDD_target_s2_comp * (2 * np.pi * speed_of_light**2 * d_m_opt**2 * cos_theta_d_analytic_val**3) / \
                         (lambda_0_m_opt**3 * m_order_comp**2)
        L_eff_m_gdd_only = L_g_m_analytic / N_passes_comp
    
    if verbose:
        print(f"Angle initial (Littrow m={m_order_comp}) : {theta_i_deg_gdd_only:.2f} deg")
        print(f"Distance calculée : {L_eff_m_gdd_only * 100:.2f} cm")

    if optimization_level == 'GDD_only':
        return L_eff_m_gdd_only, theta_i_deg_gdd_only

    # --- Partie 2 : Optimisation Numérique (GDD et TOD Cibles) ---
    if verbose: print("\n--- Optimisation Partie 2 : GDD et TOD Cibles (Numérique) ---")
    
    # pulse_ref_for_calc est utilisé pour appeler calculate_grating_dispersion
    # Il doit avoir les bonnes propriétés (comme center_wavelength_nm)
    pulse_ref_for_calc = pulse_stretched_input.create_cloned_pulse() 

    def cost_function_disp_coeffs(params, pulse_ref, target_gdd, target_tod):
        L_eff_m, theta_i_deg = params
        gdd, tod, _ = pulse_ref.calculate_grating_dispersion(
            grating_lines_per_mm, L_eff_m, theta_i_deg, m_order_comp, N_passes_comp)
        if np.isinf(gdd): return 1e12
        err_gdd = ((gdd - target_gdd) / target_gdd)**2 if target_gdd != 0 else (gdd/0.01)**2
        err_tod = ((tod - target_tod) / target_tod)**2 if target_tod != 0 else (tod/0.001)**2
        return err_gdd + err_tod

    initial_guess_disp = [L_eff_m_gdd_only, theta_i_deg_gdd_only]
    bounds_opt = [(0.001, 2.0), (-88.0, -10.0)] # Angle doit permettre diffraction pour m=-1
    optim_options_disp = {'ftol': 1e-12, 'gtol': 1e-8}

    result_disp = minimize(
        cost_function_disp_coeffs, initial_guess_disp, 
        args=(pulse_ref_for_calc, target_gdd_comp, target_tod_comp),
        method='L-BFGS-B', bounds=bounds_opt, options=optim_options_disp)

    L_eff_m_disp_opt, theta_i_deg_disp_opt = result_disp.x
    if verbose:
        print(f"Optimisation (GDD/TOD Cibles) terminée. Coût: {result_disp.fun:.2e}")
        print(f"  Angle : {theta_i_deg_disp_opt:.3f} deg, Distance : {L_eff_m_disp_opt * 100:.3f} cm")

    if optimization_level == 'GDD_TOD_coeffs':
        return L_eff_m_disp_opt, theta_i_deg_disp_opt

    # --- Partie 3 : Optimisation Numérique (Maximisation de la Puissance Crête) ---
    if verbose: print("\n--- Optimisation Partie 3 : Maximisation Puissance Crête (Numérique) ---")

    def cost_function_peak_power(params, pulse_to_recompress_ref):
        L_eff_m, theta_i_deg = params
        temp_pulse = pulse_to_recompress_ref.create_cloned_pulse()
        gdd_comp_pp, _, _ = temp_pulse.apply_grating_compressor( # Applique GDD, TOD, FOD
            grating_lines_per_mm, L_eff_m, theta_i_deg, m_order_comp, N_passes_comp)
        if np.isinf(gdd_comp_pp): return 1e12 
        peak_power = np.max(temp_pulse.it)
        return -peak_power

    initial_guess_peak_power = [L_eff_m_disp_opt, theta_i_deg_disp_opt]
    optim_options_peak_power = {'ftol': 1e-9, 'gtol': 1e-6, 'eps':1e-9}

    result_peak_power = minimize(
        cost_function_peak_power, initial_guess_peak_power, 
        args=(pulse_stretched_input,), 
        method='L-BFGS-B', bounds=bounds_opt, options=optim_options_peak_power)

    L_eff_m_final, theta_i_deg_final = result_peak_power.x
    if verbose:
        print(f"Optimisation (Maximisation Puissance Crête) terminée.")
        print(f"  Succès : {result_peak_power.success}, Message : {result_peak_power.message}")
        print(f"  Valeur finale de la fonction de coût (-Puissance Crête) : {result_peak_power.fun:.3e} (W)")
        print(f"  Puissance crête maximale atteinte : {-result_peak_power.fun:.3e} (W)")
        print(f"Angle final : {theta_i_deg_final:.3f} deg, Distance finale : {L_eff_m_final * 100:.3f} cm")
              
    return L_eff_m_final, theta_i_deg_final

# ==============================================================================
# === Script Principal d'Exemple ===============================================
# ==============================================================================
if __name__ == '__main__':
    verbose_main = True # Contrôle l'affichage dans la fonction et ici

    # Paramètres de l'impulsion et de l'étireur
    center_wavelength_nm_main = 1550.0
    fwhm_ps_main = 0.150
    energy_J_main = 100e-12

    GDD_applied_by_stretcher_ps2 = 28.96
    TOD_applied_by_stretcher_ps3 = -1.063
    FOD_applied_by_stretcher_ps4 = 0.00 

    # Paramètres du compresseur
    grating_lines_per_mm_main = 1200.5
    m_order_main = -1
    N_passes_main = 2

    # Création de l'impulsion initiale
    pulse_initial_main = Pulse(
        pulse_type='sech',
        center_wavelength_nm=center_wavelength_nm_main,
        fwhm_ps=fwhm_ps_main,
        time_window_ps=2550.0, # Fenêtre plus petite pour tests rapides de l'exemple
        npts=2**18,          # Moins de points pour tests rapides
        epp=energy_J_main
    )
    
    # Création de l'impulsion étirée
    pulse_stretched_main = pulse_initial_main.create_cloned_pulse()
    pulse_stretched_main.chirp_pulse_W(
        GDD=GDD_applied_by_stretcher_ps2, 
        TOD=TOD_applied_by_stretcher_ps3, 
        FOD=FOD_applied_by_stretcher_ps4 
    )
    if verbose_main: print("Impulsion initiale et étirée créées.")

    # --- Test avec le niveau d'optimisation 'peak_power' ---
    chosen_level_main = 'peak_power' 
    if verbose_main: print(f"\n\n=== TEST DE L'OPTIMISATION '{chosen_level_main}' ===")

    L_opt_final, theta_opt_final = optimize_compressor(
        pulse_stretched_input=pulse_stretched_main,
        grating_lines_per_mm=grating_lines_per_mm_main,
        m_order_comp=m_order_main,
        N_passes_comp=N_passes_main,
        optimization_level=chosen_level_main,
        verbose=verbose_main
    )

    if verbose_main:
        print("\n=== PARAMÈTRES OPTIMAUX FINAUX RETOURNÉS PAR LA FONCTION ===")
        print(f"L_eff = {L_opt_final*100:.3f} cm, Theta_i = {theta_opt_final:.3f} deg")

    # Appliquer ces paramètres pour obtenir l'impulsion recompressée
    pulse_recompressed_main = pulse_stretched_main.create_cloned_pulse()
    g_comp_final, t_comp_final, f_comp_final = pulse_recompressed_main.apply_grating_compressor(
                                grating_lines_per_mm_main, L_opt_final, theta_opt_final, 
                                m_order_main, N_passes_main)

    if verbose_main:
        print(f"\nDispersion du compresseur avec paramètres optimaux ({chosen_level_main}):")
        print(f"  GDD_comp = {g_comp_final:.4f} ps^2, TOD_comp = {t_comp_final:.4f} ps^3, FOD_comp = {f_comp_final:.4f} ps^4")
        
        final_residual_gdd = GDD_applied_by_stretcher_ps2 + g_comp_final
        final_residual_tod = TOD_applied_by_stretcher_ps3 + t_comp_final
        final_residual_fod = FOD_applied_by_stretcher_ps4 + f_comp_final # Utilise le FOD de l'étireur défini
        print(f"Dispersion Résiduelle Finale (approx):")
        print(f"  GDD_res = {final_residual_gdd:.2e} ps^2")
        print(f"  TOD_res = {final_residual_tod:.2e} ps^3")
        print(f"  FOD_res = {final_residual_fod:.2e} ps^4")

        print(f"FWHM finale = {pulse_recompressed_main.calc_width()*1000:.2f} fs")
        print(f"Puissance crête finale = {np.max(pulse_recompressed_main.it):.3e} W")

    # --- Visualisation des résultats du test ---
    fig_test, ax_test = plt.subplots(figsize=(10, 6))
    ax_test.plot(pulse_initial_main.t_ps, pulse_initial_main.it, 'k-', label=f'Initiale (FWHM = {pulse_initial_main.calc_width()*1000:.1f} fs)')
    ax_test.plot(pulse_recompressed_main.t_ps, pulse_recompressed_main.it, 'r-', lw=2, label=f'Opt. "{chosen_level_main}" (FWHM = {pulse_recompressed_main.calc_width()*1000:.1f} fs)')
    ax_test.set_title(f'Test de la fonction optimize_compressor (Niveau: {chosen_level_main})')
    ax_test.set_xlabel('Temps (ps)')
    ax_test.set_ylabel('Puissance (W)')
    ax_test.set_xlim(-1.0, 1.0) 
    ax_test.legend()
    ax_test.grid(True)
    plt.tight_layout()
    plt.show()