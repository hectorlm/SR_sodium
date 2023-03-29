import ipywidgets as ipyw
import numpy as np
import matplotlib.pyplot as plt


class ImageSliceViewer3D:
    """
    ImageSliceViewer3D is for viewing volumetric image slices in jupyter or
    ipython notebooks.

    User can interactively change the slice plane selection for the image and
    the slice plane being viewed.

    Argumentss:
    Volume = 3D input image
    figsize = default(8,8), to set the size of the figure
    cmap = default('plasma'), string for the matplotlib colormap. You can find
    more matplotlib colormaps on the following link:
    https://matplotlib.org/users/colormaps.html

    """

    def __init__(self, volume1, volume2, figsize=(16, 16), cmap='viridis'):
        self.volume1 = volume1
        self.volume2 = volume2
        self.figsize = figsize
        self.cmap = cmap
        self.v1 = [np.min(volume1), np.max(volume1)]
        self.v2 = [np.min(volume2), np.max(volume2)]
        self.v = [np.minimum(self.v1[0], self.v2[0]),
                  np.maximum(self.v1[1], self.v2[1])]
        self.fig = None
        self.axs = None
        self.vol1 = None
        self.vol2 = None
        self.vol3 = None
        self.vol4 = None

        # Call to select slice plane
        ipyw.interact(self.view_selection, view=ipyw.RadioButtons(
            options=['x-y', 'y-z', 'z-x'], value='y-z',
            description='Slice plane selection:', disabled=False,
            style={'description_width': 'initial'}))

    def view_selection(self, view):
        # Transpose the volume to orient according to the slice plane selection
        orient = {"y-z": [1, 2, 0], "z-x": [2, 0, 1], "x-y": [0, 1, 2]}
        self.vol1 = np.transpose(self.volume1, orient[view])
        self.vol2 = np.transpose(self.volume2, orient[view])
        maxz = self.vol1.shape[2] - 1

        # Call to view a slice within the selected slice plane
        ipyw.interact(self.plot_slice,
                      z=ipyw.IntSlider(min=0, max=maxz, step=1, continuous_update=False,
                                       description='Image Slice:'))

    def plot_slice(self, z):
        # Plot slice for the given plane and slice
        self.fig, self.axs = plt.subplots(1, 2, figsize=self.figsize)
        self.axs[0, 0].imshow(self.vol1[:, :, z], cmap=plt.get_cmap(self.cmap),
                              vmin=self.v[0], vmax=self.v[1])
        self.axs[0, 0].set_title('Low Res.')
        self.axs[0, 0].axis('off')
        self.axs[0, 1].imshow(self.vol2[:, :, z], cmap=plt.get_cmap(self.cmap),
                              vmin=self.v[0], vmax=self.v[1])
        self.axs[0, 1].set_title('High Res.')
        self.axs[0, 1].axis('off')
        # self.axs[1, 0].imshow(self.vol3[:, :, z], cmap=plt.get_cmap(self.cmap),
        #                       vmin=self.v[0], vmax=self.v[1])
        # self.axs[1, 0].set_title('Reconstructed')
        # self.axs[1, 0].axis('off')
        # self.axs[1, 1].imshow(self.vol4[:, :, z], cmap=plt.get_cmap(self.cmap),
        #                       vmin=self.v[0], vmax=self.v[1])
        # self.axs[1, 1].set_title('Error 1x')
        # self.axs[1, 1].axis('off')
