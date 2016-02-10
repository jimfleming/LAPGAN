def plot_images(images):
    fig = plt.figure(1, (3, 3))
    grid = ImageGrid(fig, 111, nrows_ncols=(3, 3), axes_pad=0.1, aspect=True)

    for i in range(9):
        grid[i].imshow(images[i])
        grid[i].grid(False)
