from matplotlib import colors
cmap = colors.ListedColormap(['#009900',
                                  '#c6b044',
                                  '#fbff13',
                                  '#b6ff05',
                                  '#27ff87',
                                  '#1c0dff',
                                  '#a5a5a5',
                                  '#69fff8',
                                  '#f9ffa4',
                                  '#1c0dff',
                                  '#ffffff'])

# folder path
image_path = "/home/tortes/Downloads/tar/10-19/s2"
elevation_path = "/home/tortes/Downloads/tar/10-19/ele"
DEM_path = "/home/tortes/Downloads/tar/10-19/dem"
pkl_path = "/home/tortes/gittmp/dfc2020_baseline/code/10-13/args.pkl"
model_path = "/home/tortes/gittmp/dfc2020_baseline/code/10-13/checkpoint_step_10000.pth"
output_path = "/home/tortes/gittmp/dfc2020_baseline/terrain/output_image"

# image factor
band_true = [4,3,2]
brightness_factor = 3
ppi_size = 256

# terrain
slope_threshold = 8.0

# land cover
band_choose = [2,3,4,8]
classnames = ["Forest", "Shrubland", "Savanna", "Grassland", "Wetlands",
            "Croplands", "Urban/Built-up", "Snow/Ice", "Barren", "Water"]