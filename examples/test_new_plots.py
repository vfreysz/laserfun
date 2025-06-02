"""
================================
Example: Plotting Pulse Profiles
================================

This script demonstrates how to use the `plot_temporal` and `plot_spectral`
methods of the `Pulse` object to visualize laser pulses.
"""

import numpy as np
import matplotlib.pyplot as plt
# from scipy.constants import c # c_nmps is defined in laserfun.pulse

# This assumes 'laserfun' is installed or in the python path
# and that pulse.py now contains the new plotting methods.
from laserfun import Pulse
from laserfun.pulse import c_nmps # Import c_nmps if needed and available, or define locally

# 1. --- Create a sample pulse ---
# A non-zero GDD is used to add chirp for a meaningful phase plot.
pulse = Pulse(pulse_type='gaussian',
              npts=2**14,
              time_window_ps=20.0,
              fwhm_ps=0.5,
              center_wavelength_nm=1030.0,
              power=10.0,  # Example: 10 W peak power
              GDD=-0.01)   # Example: Group Delay Dispersion in ps^2


# 2. --- Simple plotting ---
# Methods will now create their own figures with a default size.
print("Demonstrating simple, standalone plots...")

# Plot temporal profile (with phase by default)
fig_temporal, ax_temporal = pulse.plot_temporal()
ax_temporal.set_xlim(-5, 5) # Custom xlim can still be applied

# Plot spectral profile vs. frequency
fig_spectral_freq, ax_spectral_freq = pulse.plot_spectral(x_axis='frequency')
center_freq = pulse.centerfrequency_THz
ax_spectral_freq.set_xlim(center_freq - 2.0 * pulse._get_spectral_fwhm(pulse.f_THz, np.abs(pulse.aw)**2), 
                          center_freq + 2.0 * pulse._get_spectral_fwhm(pulse.f_THz, np.abs(pulse.aw)**2)) # Dynamic xlim based on FWHM


# 3. --- Advanced plotting on a grid ---
print("Demonstrating advanced plotting on a grid of subplots...")

# User creates figure and axes grid here
fig_advanced, (ax1_adv, ax2_adv) = plt.subplots(1, 2, figsize=(14, 6))

# Pass the created axes to the plotting methods
pulse.plot_temporal(show_phase=False, ax=ax1_adv)
pulse.plot_spectral(x_axis='wavelength', ax=ax2_adv)

# Further customize the subplots
ax1_adv.set_title('Temporal Profile (Intensity only)') # Override default method title if desired
ax1_adv.set_xlim(-5, 5)
ax2_adv.set_title('Spectral Profile (Wavelength)') # Override default method title if desired
ax2_adv.set_xlim(pulse.center_wavelength_nm - 20, pulse.center_wavelength_nm + 20) # Dynamic xlim

fig_advanced.suptitle('Full Pulse Visualization', fontsize=16)

# For complex figures, final tight_layout call is often best
try:
    fig_advanced.tight_layout(rect=[0, 0.03, 1, 0.95])
except Exception:
    fig_advanced.tight_layout()


# 4. --- Show all generated figures ---
plt.show()