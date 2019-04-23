import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import patches, colors
from statsmodels.sandbox.stats import multicomp
import scipy as sp
import numpy as np


cmap = matplotlib.colors.LinearSegmentedColormap.from_list('colormap', ['blue', 'lightgray', 'red'])
def plot_ellipse_values(values, ellipse_pars=None, size=(1000, 1000), 
                        vmin=None, vmax=None, cmap=plt.cm.coolwarm, ax=None, **kwargs):

    ''' values is a n-by-m array'''

    values[np.isnan(values)] = 0
    if ellipse_pars is None:
        a = 350
        b = 150
        x = 500
        y = 500

        theta = 45. / 180 * np.pi

    else:
        a, b, x, y, theta = ellipse_pars

    A = a**2 * (np.sin(theta))**2 + b**2 * (np.cos(theta))**2
    B = 2 * (b**2 - a**2) * np.sin(theta) * np.cos(theta)
    C = a**2 * np.cos(theta)**2 + b**2 * np.sin(theta)**2
    D = -2 * A * x - B* y
    E = -B * x - 2 * C * y
    F = A* x**2 + B*x*y + C*y**2 - a**2*b**2

    X,Y = np.meshgrid(np.arange(size[0]), np.arange(size[1]))

    in_ellipse = A*X**2 + B*X*Y +C*Y**2 + D*X + E*Y +F < 0

    pc1 = np.array([[np.cos(theta)], [np.sin(theta)]])
    pc2 = np.array([[np.cos(theta - np.pi/2.)], [np.sin(theta - np.pi/2.)]])

    pc1_distance = pc1.T.dot(np.array([(X - x).ravel(), (Y - y).ravel()])).reshape(X.shape)
    pc2_distance = pc2.T.dot(np.array([(X - x).ravel(), (Y - y).ravel()])).reshape(X.shape)

    pc1_quantile = np.floor((pc1_distance / a + 1 ) / 2. * values.shape[0])
    pc2_quantile = np.floor((pc2_distance / b + 1 ) / 2. * values.shape[1])

    im = np.zeros_like(X, dtype=float)

    for pc1_q in np.arange(values.shape[0]):
        for pc2_q in np.arange(values.shape[1]):
            im[in_ellipse * (pc1_quantile == pc1_q) & (pc2_quantile == pc2_q)] = values[pc1_q, pc2_q]

    im = np.ma.masked_array(im, ~in_ellipse)
#     cmap.set_bad('grey')
    if ax is None:
        cax = plt.imshow(im, origin='lower', cmap=cmap, vmin=vmin, vmax=vmax, **kwargs)
    else:
        ax.imshow(im, origin='lower', cmap=cmap, vmin=vmin, vmax=vmax, **kwargs)
        cax = ax
#    sns.despine()

    return cax

def visualize_stn_model(df, dependent_var='y', ax=None, vmin=None, vmax=None, **kwargs):
    if ax is None:
        f, ax = plt.subplots(1, 1)
    # recode... ugly
    df['pc1_sector_name'] = 'vmd_' + df['pc1_sector_number'].astype(int).astype(str)
    df['pc2_sector_name'] = 'mml_' + df['pc2_sector_number'].astype(int).astype(str)
    vmd_labels = df['pc1_sector_name'].unique()
    mml_labels = df['pc2_sector_name'].unique()
    
    unstacked = df.groupby(['pc1_sector_name', 'pc2_sector_name'])[dependent_var].mean().unstack(1).ix[vmd_labels, mml_labels]

    if vmin is None:
        vmin = np.nanpercentile(unstacked.values, 5)
    if vmax is None:
        vmax = np.nanpercentile(unstacked.values, 95)
    plot_ellipse_values(unstacked.values, ax=ax, vmin=vmin, vmax=vmax, **kwargs)
    ax.axis('off')
    return ax

def plot_intensity_across_axis(df, dependent_var='y', x_axis='pc1_mm', ax=None, **kwargs):
    if ax is None:
        f, ax = plt.subplots(1, 1)
        
    data_per_coordinate = df.groupby([x_axis])[dependent_var].mean().reset_index()
    ax.plot(data_per_coordinate[x_axis], data_per_coordinate[dependent_var], **kwargs)
    ax.set_xlabel(x_axis)
    ax.set_ylabel('Rate')
    return ax


def plot_along_projection_axis(df, ax=None, log=False, **kwargs):
    if ax is None:
        f, ax = plt.subplots(1, 1)
    
    df = df.sort_values('projection_axis')
    if log:
        df['rate'] = np.log(df['rate'])
        df['y_predicted'] = np.log(df['y_predicted'])
    ax.plot(df['projection_axis'], df['rate'], '.', label='data', **kwargs)
    ax.plot(df['projection_axis'], df['y_predicted'], linewidth=3, label='model', **kwargs)

    ax.set_xlabel('projection_axis')
    ax.set_ylabel('Rate')
    ax.legend()
    return ax

def plot_single_pc(df, type='_mm_norm'):
    """
    Plots the STN, collapsing over a dimension
    """
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    gs = gridspec.GridSpec(3, 4)
    ax_stn_data = plt.subplot(gs[0,0])
    ax_stn_model = plt.subplot(gs[0,1])
    
    ax_graph1 = plt.subplot(gs[1,0:2])
    ax_graph2 = plt.subplot(gs[0,2:4])
    ax_graph3 = plt.subplot(gs[1,2:4])
    
    proj_axis = plt.subplot(gs[2,0:2])
    residuals_ax = plt.subplot(gs[2,2:4])
    
    vmin, vmax = np.nanpercentile(df['rate'], [5, 95])

    visualize_stn_model(df, dependent_var='rate', ax=ax_stn_data, vmin=vmin, vmax=vmax)
    visualize_stn_model(df, dependent_var='y_predicted', ax=ax_stn_model, vmin=vmin, vmax=vmax)
    
    plot_intensity_across_axis(df, dependent_var='rate', ax=ax_graph1, x_axis='pc1'+type, label='Data')
    plot_intensity_across_axis(df, dependent_var='y_predicted', ax=ax_graph1, x_axis='pc1'+type, label='Model')

    plot_intensity_across_axis(df, dependent_var='rate', ax=ax_graph2, x_axis='pc2'+type, label='Data')
    plot_intensity_across_axis(df, dependent_var='y_predicted', ax=ax_graph2, x_axis='pc2'+type, label='Model')
    
    plot_intensity_across_axis(df, dependent_var='rate', ax=ax_graph3, x_axis='slice'+type, label='Data')
    plot_intensity_across_axis(df, dependent_var='y_predicted', ax=ax_graph3, x_axis='slice'+type, label='Model')
    
    if 'projection_axis' in df.columns:
        plot_along_projection_axis(df, ax=proj_axis)
    sns.kdeplot(df['residuals'], ax=residuals_ax)
    
    ax_stn_data.set_title('Data')
    ax_stn_model.set_title('Model')
    ax_graph1.legend()
    ax_graph2.legend()
    ax_graph3.legend()
    
    return plt.gcf()




def plot_per_slice(df, slices=[0,1,2,3,4,5,6,7,8,9]):
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    gs = gridspec.GridSpec(len(slices)+1, 4)

    vmin = np.nanpercentile(df['rate'], 5)
    vmax = np.nanperc
    entile(df['rate'], 95)
    for row_n, slice_id in enumerate(slices):
        df_this_slice = df.loc[df.slice_mm==df.slice_mm.unique()[slice_id]]
        ax_stn_data = plt.subplot(gs[row_n,0])
        ax_stn_model = plt.subplot(gs[row_n,1])
        ax_graph_pc1 = plt.subplot(gs[row_n,2])
        ax_graph_pc2 = plt.subplot(gs[row_n,3])

        visualize_stn_model(df_this_slice, dependent_var='rate', ax=ax_stn_data, vmin=vmin, vmax=vmax)
        visualize_stn_model(df_this_slice, dependent_var='y_predicted', ax=ax_stn_model, vmin=vmin, vmax=vmax)
        plot_intensity_across_axis(df_this_slice, dependent_var='rate', ax=ax_graph_pc1, label='Data')
        plot_intensity_across_axis(df_this_slice, dependent_var='y_predicted', ax=ax_graph_pc1, label='Model')
        plot_intensity_across_axis(df_this_slice, x_axis='pc2_sector_number', dependent_var='rate', ax=ax_graph_pc2, label='Data')
        plot_intensity_across_axis(df_this_slice, x_axis='pc2_sector_number', dependent_var='y_predicted', ax=ax_graph_pc2, label='Model')

        ax_stn_data.set_title('Data')
        ax_stn_model.set_title('Model')
        ax_graph_pc1.legend()
        ax_graph_pc1.set_ylim(vmin, vmax)
        ax_graph_pc2.legend()
        ax_graph_pc2.set_ylim(vmin, vmax)

    # some overall plots (across pc1, pc2, slice)
    plot_intensity_across_axis(df, x_axis='pc1_sector_number', dependent_var='rate', ax=plt.subplot(gs[-1,0]), label='Data')
    plot_intensity_across_axis(df, x_axis='pc1_sector_number', dependent_var='y_predicted', ax=plt.subplot(gs[-1,0]), label='Model')
    plot_intensity_across_axis(df, x_axis='pc2_sector_number', dependent_var='rate', ax=plt.subplot(gs[-1,1]), label='Data')
    plot_intensity_across_axis(df, x_axis='pc2_sector_number', dependent_var='y_predicted', ax=plt.subplot(gs[-1,1]), label='Model')
    plot_intensity_across_axis(df, x_axis='slice_mm', dependent_var='rate', ax=plt.subplot(gs[-1,2]), label='Data')
    plot_intensity_across_axis(df, x_axis='slice_mm', dependent_var='y_predicted', ax=plt.subplot(gs[-1,2]), label='Model')

    plt.gcf().set_size_inches(20, 20)
    

def plot_stns(df, y_type='rate', vmin=-1.5, vmax=1.5, vminmax=None, n_slices=5, n_sectors_per_axis=10, suptitle=None):
    if len(df['stain'].unique()) == 1:
        fig, axes = plt.subplots(nrows=1, ncols=1*n_slices)
        axes = axes[np.newaxis,:]
    else:
        fig, axes = plt.subplots(nrows=4, ncols=3*n_slices)
    
    
    pma_labels = df['slice_sector_labels'].unique().tolist()
    vmd_labels = df['pc1_sector_labels'].unique().tolist()
    mml_labels = df['pc2_sector_labels'].unique().tolist()
    
    for ii, (stain, d) in enumerate(df.groupby(['stain'])):
        column_set = int(np.floor(ii/4.))  # total number of columns required is n_stains / 4 (rows)
        row_n = int((ii)%4.)
        
        if vminmax is not None:
            vmin, vmax = vminmax[stain]

        for i, (slice, d2) in enumerate(d.groupby('slice_sector_labels')):
            print(row_n, pma_labels[::-1].index(slice) + n_slices*(column_set), end=' ')
            ax = axes[row_n, pma_labels.index(slice) + n_slices*(column_set)]

            n = d2.groupby(['pc1_sector_labels', 'pc2_sector_labels'])[y_type].apply(lambda v: len(v)).unstack(1).ix[vmd_labels, mml_labels] #['ventral', 'middle', 'dorsal'], ['medial', 'middle', 'lateral']]
            t = d2.groupby(['pc1_sector_labels', 'pc2_sector_labels'])[y_type].apply(lambda v: sp.stats.ttest_1samp(v, 0, nan_policy='omit')[0]).unstack(1).ix[vmd_labels, mml_labels]#['ventral', 'middle', 'dorsal'], ['medial', 'middle', 'lateral']]
            p = d2.groupby(['pc1_sector_labels', 'pc2_sector_labels'])[y_type].apply(lambda v: sp.stats.ttest_1samp(v, 0, nan_policy='omit')[1]).unstack(1).ix[vmd_labels, mml_labels]
            mean = d2.groupby(['pc1_sector_labels', 'pc2_sector_labels'])[y_type].mean().unstack(1).ix[vmd_labels, mml_labels]

            # FDR: as we are doing 27 seperate t-tests we need to correct for multiple comparisons:
            p.values[:] = multicomp.fdrcorrection0(p.values.ravel())[1].reshape(n_sectors_per_axis, n_sectors_per_axis)

            # Providing some parameters for plotting the figures
            if i == len(d.groupby('slice_5'))/2:
                a, b, x, y, theta  = 350, 150, 300, 275, 45
            else:
                a, b, x, y, theta  = 300, 125, 300, 275, 45.

#             vmin, vmax = np.percentile(mean.values, [2.5, 97.5])
#            vmax = np.percentile(t.values, 2.5)
    #         plot_ellipse_values(t[p<0.05].values, size=(600, 550), ellipse_pars=(a, b, x, y,  theta / 180. * np.pi), vmin=-7, vmax=7, cmap=cmap, ax=ax)
            plot_ellipse_values(mean.values, size=(600, 550), ellipse_pars=(a, b, x, y,  theta / 180. * np.pi), vmin=vmin, vmax=vmax, cmap=cmap, ax=ax)

            e1 = patches.Ellipse((x, y), a*2, b*2,
                                 angle=theta, linewidth=2, fill=False, zorder=2)

            ax.add_patch(e1)
            ax.set_xticks([])
            ax.set_yticks([])

            sns.despine(bottom=True, left=True, right=True)

            if slice == pma_labels[int(n_slices/2)]: #'middle':
                ax.set_title(stain, fontsize=24)

    #             print stain
    #             print p.values
    fig.set_size_inches(10.*2, 4.*2)
    
    if suptitle is not None:
        htop = .9
        fig.suptitle(suptitle)
    else:
        htop = 1
        
    fig.subplots_adjust(hspace=.275, wspace=0.00, bottom=0.01, left=0.0, top=htop, right=1)

    # outer boxes
    plt.plot([0, 0], [0, htop], color='black', lw=1, transform=plt.gcf().transFigure, clip_on=False)
    plt.plot([0, 1], [htop, htop], color='black', lw=1, transform=plt.gcf().transFigure, clip_on=False)
    plt.plot([1, 1], [0, htop], color='black', lw=1, transform=plt.gcf().transFigure, clip_on=False)
    plt.plot([0, 1], [0, 0], color='black', lw=1, transform=plt.gcf().transFigure, clip_on=False)

    # lines at 1/3rd, 2/3rd
    plt.plot([1/3., 1/3.], [0, htop], color='black', lw=1, transform=plt.gcf().transFigure, clip_on=False)
    plt.plot([2/3., 2/3.], [0, htop], color='black', lw=1, transform=plt.gcf().transFigure, clip_on=False)
