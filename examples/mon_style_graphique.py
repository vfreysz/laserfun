# mon_style_graphique.py

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch

FONT_NAME = 'Eurostile'

def initialiser_police():
    try:
        plt.rcParams['font.family'] = FONT_NAME
        print(f"Police {FONT_NAME} configurée avec succès pour Matplotlib.")
    except RuntimeError:
        print(f"ERREUR : Police {FONT_NAME} non trouvée. Utilisation d'une police par défaut.")
        plt.rcParams['font.family'] = 'sans-serif'

COLOR_PALETTE = {
    'bleu_ceruleen': '#006583', 'bleu_mer': '#99C1CD', 'bleu_sherpa': '#003D4F',
    'vert': '#A6CA56', 'gris_moyen': '#DADADA', 'gris_fonce': '#2D2D2D',
    'blanc': '#FFFFFF', 'noir': '#000000',
}
FONT_SIZES = {'label': 20, 'legend': 18, 'tick': 18, 'text_annot': 17, 'title': 22}
YLIM_RANGE_DEFAULT = (0, 4)
GRADIENT_START_Y_DEFAULT = 2.5
FIXED_FIGSIZE = (10, 7)
TEXT_ANNOT_POS = {'x': 0.97, 'y1': 0.28, 'y2': 0.18}


def get_ciel_matinal_final_params():
    cp = COLOR_PALETTE
    gradient_colors_spec = {
        'bottom_color': cp['blanc'],
        'transition_start_color': cp['blanc'],
        'top_color': cp['bleu_mer']
    }
    return {
        'plot_title_default': "Ciel Matinal Finalisé",
        'fig_bg_color': cp['blanc'],
        'ax_bg_color': cp['blanc'],
        'gradient_fill_colors': gradient_colors_spec,
        'gradient_start_y_abs': GRADIENT_START_Y_DEFAULT,
        'line_color': cp['bleu_sherpa'],
        'line_width': 2,
        'text_color': cp['noir'],
        'axis_color': cp['noir'],
        'tick_color': cp['noir'],
        'spine_linewidth': 1.2,
        'legend_frameon': False,
        'text_annot_bbox_visible': False,
        'grid_params': {'visible': True, 'horizontal_only': False,
                        'color': cp['gris_moyen'], 'linestyle': '-',
                        'linewidth': 0.5, 'alpha': 0.5},
        'ylim_range_default': YLIM_RANGE_DEFAULT,
        'font_sizes': FONT_SIZES
    }

def appliquer_style_base_axes(ax, style_params):
    cp = COLOR_PALETTE
    ax.set_facecolor(style_params.get('ax_bg_color', cp['blanc']))
    # S'assurer que le fond de l'axe est bien derrière les autres éléments
    ax.patch.set_zorder(0.1) 

    grid_params = style_params.get('grid_params', {})
    grid_visible = grid_params.get('visible', True)
    if grid_visible:
        grid_color = grid_params.get('color', cp['gris_moyen'])
        grid_linestyle = grid_params.get('linestyle', '-')
        grid_linewidth = grid_params.get('linewidth', 0.5)
        grid_alpha = grid_params.get('alpha', 0.5)
        # Le zorder de la grille doit être supérieur à celui du patch de l'axe
        current_grid_zorder = grid_params.get('zorder', 0.5) 
        if grid_params.get('horizontal_only', False):
            ax.yaxis.grid(True, color=grid_color, linestyle=grid_linestyle,
                          linewidth=grid_linewidth, alpha=grid_alpha, zorder=current_grid_zorder)
            ax.xaxis.grid(False)
        else:
            ax.grid(True, color=grid_color, linestyle=grid_linestyle,
                    linewidth=grid_linewidth, alpha=grid_alpha, zorder=current_grid_zorder)
    else:
        ax.grid(False)

    current_font_sizes = style_params.get('font_sizes', FONT_SIZES)
    tick_color = style_params.get('tick_color', cp['noir'])
    ax.tick_params(axis='x', colors=tick_color, labelsize=current_font_sizes['tick'], width=1.2, labelrotation=0, zorder=2.0)
    ax.tick_params(axis='y', colors=tick_color, labelsize=current_font_sizes['tick'], width=1.2, zorder=2.0)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontname(FONT_NAME)
        label.set_color(tick_color)

    axis_color = style_params.get('axis_color', cp['noir'])
    spine_linewidth = style_params.get('spine_linewidth', 1.2)
    ax.spines['bottom'].set_color(axis_color)
    ax.spines['left'].set_color(axis_color)
    ax.spines['bottom'].set_linewidth(spine_linewidth)
    ax.spines['left'].set_linewidth(spine_linewidth)
    ax.spines['top'].set_visible(style_params.get('spine_top_visible', False))
    ax.spines['right'].set_visible(style_params.get('spine_right_visible', False))
    ax.spines['bottom'].set_zorder(2.0)
    ax.spines['left'].set_zorder(2.0)

    ax.xaxis.label.set_color(style_params.get('text_color', cp['noir']))
    ax.xaxis.label.set_fontsize(current_font_sizes['label'])
    ax.xaxis.label.set_fontweight('bold')
    ax.xaxis.label.set_fontname(FONT_NAME)
    
    ax.yaxis.label.set_color(style_params.get('text_color', cp['noir']))
    ax.yaxis.label.set_fontsize(current_font_sizes['label'])
    ax.yaxis.label.set_fontweight('bold')
    ax.yaxis.label.set_fontname(FONT_NAME)

def ajouter_remplissage_degrade(ax, x_data, y_data, style_params):
    if 'gradient_fill_colors' not in style_params or not style_params['gradient_fill_colors']:
        return
    if len(x_data) < 2 or len(y_data) < 2:
        print("Note: Pas assez de points pour le remplissage dégradé.")
        return

    grad_colors_spec = style_params['gradient_fill_colors']
    grad_start_y_abs = style_params.get('gradient_start_y_abs', 0.0) 

    current_xlim = ax.get_xlim()
    current_ylim = ax.get_ylim() 

    if current_ylim[1] <= current_ylim[0]: 
        print("Attention: Limites Y de l'axe invalides pour le dégradé.")
        return

    stop_point_transition_start_rel = (grad_start_y_abs - current_ylim[0]) / (current_ylim[1] - current_ylim[0])
    stop_point_transition_start_rel = np.clip(stop_point_transition_start_rel, 0.0, 1.0)

    cmap_fill = LinearSegmentedColormap.from_list(
        "custom_fill_gradient",
        [(0, grad_colors_spec['bottom_color']),
         (stop_point_transition_start_rel, grad_colors_spec['transition_start_color']),
         (1.0, grad_colors_spec['top_color'])]
    )

    gradient_img_array = np.linspace(0, 1, 256)
    gradient_img_array = np.vstack((gradient_img_array, gradient_img_array)).T

    im = ax.imshow(gradient_img_array, aspect='auto',
                   extent=[*current_xlim, *current_ylim], 
                   origin='lower', cmap=cmap_fill,
                   zorder=0.7) # zorder au-dessus du patch (0.1) et grille (0.5) mais sous la ligne

    path_vertices = [(x_data[0], current_ylim[0])] 
    path_vertices.extend(zip(x_data, y_data))
    path_vertices.append((x_data[-1], current_ylim[0])) 
    path_vertices.append((x_data[0], current_ylim[0])) 

    clip_path = MplPath(path_vertices)
    patch = PathPatch(clip_path, transform=ax.transData)
    im.set_clip_path(patch)

def styliser_legende(legend, style_params):
    cp = COLOR_PALETTE
    current_font_sizes = style_params.get('font_sizes', FONT_SIZES)
    main_text_color = style_params.get('text_color', cp['noir'])
    
    for text_legend_item in legend.get_texts():
        text_legend_item.set_color(main_text_color)
        text_legend_item.set_fontname(FONT_NAME)
        text_legend_item.set_fontweight('bold') # Homogénéisation avec FWHM
        text_legend_item.set_fontsize(current_font_sizes['legend']) # Homogénéisation avec FWHM
    if legend.get_title() is not None:
        legend.get_title().set_fontname(FONT_NAME)
        legend.get_title().set_color(main_text_color)
        legend.get_title().set_fontsize(current_font_sizes['legend'])
        legend.get_title().set_fontweight('bold')
    
    if not style_params.get('legend_frameon', True) and legend.get_frame() is not None:
        legend.get_frame().set_alpha(0) 
        legend.get_frame().set_edgecolor('none')

def ajouter_annotations_stylisees(ax, x_rel, y_rel, texte, style_params):
    cp = COLOR_PALETTE
    current_font_sizes = style_params.get('font_sizes', FONT_SIZES)
    main_text_color = style_params.get('text_color', cp['noir'])
    bbox_style = None 
    if style_params.get('text_annot_bbox_visible', False): 
         bbox_style = dict(boxstyle='round,pad=0.2',
                           fc=style_params.get('text_bbox_face_color', cp['blanc']),
                           ec=style_params.get('text_bbox_edge_color', cp['gris_moyen']),
                           alpha=0.85) 
    
    ax.text(x_rel, y_rel, texte,
            transform=ax.transAxes, fontsize=current_font_sizes['text_annot'],
            color=main_text_color,
            verticalalignment='top', horizontalalignment='right', fontweight='bold', fontname=FONT_NAME,
            bbox=bbox_style, zorder=2.1)