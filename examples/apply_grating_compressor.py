import laserfun as lf
import numpy as np # Importer numpy ici aussi pour l'exemple

# Créer une impulsion
mon_impulsion = lf.Pulse(pulse_type='gaussian',
                         center_wavelength_nm=1030,
                         fwhm_ps=0.2,
                         epp=1e-9)

print(f"Durée FWHM initiale : {mon_impulsion.calc_width():.4f} ps")
# Sauvegarder la phase spectrale initiale pour comparaison si besoin
phase_initiale = np.angle(mon_impulsion.aw)


# Paramètres du compresseur
traits_par_mm = 1700
ordre_diffraction = -1
angle_incidence_deg = 72.5 # Angle proche de Littrow pour 1030nm et 1700 traits/mm
L_efficace_metres = 0.05 # 5 cm, exemple
N_passes = 2

try:
    mon_impulsion.apply_grating_compressor(
        grating_lines_per_mm=traits_par_mm,
        m=ordre_diffraction,
        L_eff_m=L_efficace_metres,
        theta_i_deg=angle_incidence_deg,
        N_passes_formula=N_passes
    )
    print(f"Durée FWHM après compresseur : {mon_impulsion.calc_width():.4f} ps")
    phase_finale = np.angle(mon_impulsion.aw)

    # Vérifier que la phase a changé
    assert not np.allclose(phase_initiale, phase_finale), "La phase spectrale n'a pas changé !"
    print("La nouvelle méthode semble fonctionner et a modifié la phase de l'impulsion.")

except ValueError as e:
    print(f"Erreur lors du test : {e}")

# Optionnel : ajouter des tests unitaires plus formels dans laserfun/tests/test_all.py
# pour cette nouvelle fonctionnalité.