from matplotlib import pyplot as plt
import matplotlib


def show_img(img):
    """
    Show an image with matplotlib
    Args:
        img: image to show

    Returns: None
    """

    # Some DPI Scaling ..
    dpi = matplotlib.rcParams['figure.dpi'] / 1.5
    # Determine the figures size in inches to fit your image
    height, width, depth = img.shape
    figsize = width / float(dpi), height / float(dpi)
    plt.figure(figsize=figsize)
    # Some color shifting
    plt.imshow(img[..., ::-1])  # RGB-> BGR
    # plt.imshow(img)
    plt.xticks([]), plt.yticks([])
    plt.show()


def show_img_grid(imgs, cols=2, rows=1, figsize=(20, 20)):
    """
    Helper function to show images in a grid
    Args:
        imgs: list of images
        cols: number of columns
        rows: number of rows
        figsize: figure size

    Returns: None
    """

    fig, ax = plt.subplots(nrows=rows, ncols=cols, figsize=figsize)
    for i, axi in enumerate(ax.flat):
        axi.imshow(imgs[i])
        axi.set(xticks=[], yticks=[])
    plt.show()
