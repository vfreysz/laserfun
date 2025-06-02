"""
====================================================================
Example: Plotting Pulse with Custom Styling from mon_style_graphique
====================================================================
This script demonstrates applying detailed custom styles.
"""

import numpy as np
import matplotlib.pyplot as plt

# Import your custom styling module
import mon_style_graphique as msg

# Import the Pulse object from the laserfun library
from laserfun import Pulse
try:
    from laserfun.pulse import c_nmps # c_nmps is used by plot_spectral's FWHM conversion
except ImportError:
    c_mks_local = 299792458.0
    c_nmps = c_mks_local * 1e9 / 1e12


# --- STEP 0: Initialize global font ---
msg.initialiser_police()
plt.rcParams['axes.unicode_minus'] = False

# --- STEP 1: Create a sample Pulse object ---
pulse = Pulse(pulse_type='gaussian',
              npts=2**14,
              time_window_ps=20.0,
              fwhm_ps=0.5,
              center_wavelength_nm=1030.0,
              power=10.0,
              GDD=-0.01)

# --- STEP 2: Define style_params dictionary ---
style_params = msg.get_ciel_matinal_final_params()

# Override or add specific style elements for lines if needed
style_params['STYLE_PROFIL_PRINCIPAL'] = {
    'color': msg.COLOR_PALETTE.get('bleu_sherpa', '#003D4F'),
    'linewidth': 2.5,
}
style_params['STYLE_PHASE'] = {
    'color': msg.COLOR_PALETTE.get('vert', '#A6CA56'),
    'linewidth': 2.0,
    'linestyle': '--',
}
# Ensure other necessary params from get_ciel_matinal_final_params are used, like:
# style_params['legend_loc'] = 'upper right' (already in get_ciel_matinal_final_params)
# style_params['fwhm_precision_1'] = True (custom, not in get_ciel_matinal_final_params)
# style_params['figure_background_transparent'] = True (already in get_ciel_matinal_final_params)
# style_params['font_sizes'] = msg.FONT_SIZES (already in get_ciel_matinal_final_params)


# --- STEP 3: Generate the basic plot ---
print("Generating temporal plot...")
fig_temporal, ax_temporal = pulse.plot_temporal(show_phase=True)

# --- STEP 4: Apply custom styles from mon_style_graphique.py ---
print("Applying custom styles...")

# 4.1. Style the plotted lines
line_intensity = ax_temporal.get_lines()[0]
line_intensity.set_color(style_params['STYLE_PROFIL_PRINCIPAL']['color'])
line_intensity.set_linewidth(style_params['STYLE_PROFIL_PRINCIPAL']['linewidth'])
line_intensity.set_zorder(1.0)

ax_temporal_phase = None
for ax_twin_candidate in fig_temporal.get_axes():
    if ax_twin_candidate != ax_temporal and not ax_twin_candidate.patch.get_visible():
        ax_temporal_phase = ax_twin_candidate
        break

line_phase = None
if ax_temporal_phase and len(ax_temporal_phase.get_lines()) > 0:
    line_phase = ax_temporal_phase.get_lines()[0]
    line_phase.set_color(style_params['STYLE_PHASE']['color'])
    line_phase.set_linewidth(style_params['STYLE_PHASE']['linewidth'])
    line_phase.set_linestyle(style_params['STYLE_PHASE']['linestyle'])

# 4.2. Apply gradient fill
ylim_to_set = style_params.get('ylim_range_default', None)
if ylim_to_set:
    ax_temporal.set_ylim(ylim_to_set)
else:
    current_y_max = np.max(line_intensity.get_ydata())
    ax_temporal.set_ylim(0, current_y_max * 1.1 if current_y_max > 0 else 1)

if 'gradient_fill_colors' in style_params and 'gradient_start_y_abs' in style_params:
    intensity_data_norm = line_intensity.get_ydata()
    time_data_ps = line_intensity.get_xdata()
    msg.ajouter_remplissage_degrade(ax_temporal, time_data_ps, intensity_data_norm,
                                    style_params)
else:
    print("Note: Gradient parameters ('gradient_fill_colors' or 'gradient_start_y_abs') missing in style_params. Skipping gradient.")

# 4.3. Style axes
msg.appliquer_style_base_axes(ax_temporal, style_params)
if ax_temporal_phase:
    msg.appliquer_style_base_axes(ax_temporal_phase, style_params)
    ax_temporal_phase.xaxis.label.set_visible(False)
    ax_temporal_phase.tick_params(axis='x', labelbottom=False, labeltop=False)
    if not style_params.get('spines_settings', {}).get('bottom', True):
         ax_temporal_phase.spines['bottom'].set_visible(False)

# 4.4. Re-create and Style the legend
handles_styled = [line_intensity]
labels_styled = [line_intensity.get_label()]

fwhm_ps_val = pulse.calc_width()
if not np.isnan(fwhm_ps_val) and fwhm_ps_val > 0:
    fwhm_text_styled = f"FWHM: {fwhm_ps_val:.1f} ps" if style_params.get('fwhm_precision_1', True) else f"FWHM = {fwhm_ps_val:.2f} ps"
    line_fwhm_dummy_for_legend, = ax_temporal.plot([],[], color='none', linestyle='None', label=fwhm_text_styled)
    handles_styled.append(line_fwhm_dummy_for_legend)
    labels_styled.append(fwhm_text_styled)

if ax_temporal_phase and line_phase:
    handles_styled.append(line_phase)
    labels_styled.append(line_phase.get_label())

legend_target_ax = ax_temporal_phase if ax_temporal_phase else ax_temporal
current_legend_obj = legend_target_ax.get_legend()
if current_legend_obj:
    current_legend_obj.remove()

new_legend = legend_target_ax.legend(handles_styled, labels_styled,
                                     loc=style_params.get('legend_loc', 'upper right'),
                                     frameon=style_params.get('legend_frameon', False),
                                     fontsize=style_params['font_sizes']['legend'])
msg.styliser_legende(new_legend, style_params)

# 4.5. Figure background and title
if style_params.get('figure_background_transparent', False):
    fig_temporal.patch.set_alpha(0)

# CORRECTED LINE: Use msg.FONT_NAME directly
ax_temporal.set_title("Temporal Profile (Custom Style)",
                      color=style_params.get('text_color'),
                      fontsize=style_params['font_sizes']['title'],
                      fontname=msg.FONT_NAME, # Changed from style_params['font_name']
                      fontweight='bold')

# 4.6. Final layout adjustments and limits
ax_temporal.set_xlim(-2.5, 2.5)
ax_temporal.set_ylim(-0, 1.1)
if ax_temporal_phase and line_phase:
    phase_y_data = line_phase.get_ydata()
    if len(phase_y_data) > 0:
        max_abs_phase_val = np.max(np.abs(phase_y_data))
        if max_abs_phase_val > 0:
            ax_temporal_phase.set_ylim(-max_abs_phase_val * 1.2, max_abs_phase_val * 1.2)

try:
    fig_temporal.tight_layout()
except Exception:
    pass

# --- STEP 5: Show and/or save the plot ---
plt.savefig("styled_temporal_plot_final_v3.png", dpi=300, transparent=style_params.get('figure_background_transparent', True))
print("Styled temporal plot saved as styled_temporal_plot_final_v3.png")
plt.show()