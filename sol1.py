### Author Amit Halbreich - ID:208917393 ###
import numpy as np
import imageio as io
from skimage import color
import matplotlib.pyplot as plt

GRAYSCALE = 1
RGB = 2
RGB_YIQ_TRANSFORMATION_MATRIX = np.array([[0.299, 0.587, 0.114],
                                          [0.596, -0.275, -0.321],
                                          [0.212, -0.523, 0.311]])

YIQ_RGB_TRANSFORMATION_MATRIX = np.array([[1, 0.956, 0.619],
                                          [1, -0.272, -0.647],
                                          [1, -1.106, 1.703]])

def read_image(filename, representation):
    """
    Reads an image and converts it into a given representation
    :param filename: filename of image on disk
    :param representation: 1 for greyscale and 2 for RGB
    :return: Returns the image as an np.float64 matrix normalized to [0,1]
    """
    img = io.imread(filename)
    if np.max(img.flatten()) > 1:
        img = img.astype(np.float64) / 255  # normalize the data to 0 - 1
    if representation == GRAYSCALE and len(np.shape(img)) == 3:
        img = color.rgb2gray(img)
    return img


def imdisplay(filename, representation):
    """
    Reads an image and displays it into a given representation
    :param filename: filename of image on disk
    :param representation: 1 for greyscale and 2 for RGB
    """
    inter_img = read_image(filename, representation)
    if representation == GRAYSCALE:
        plt.imshow(inter_img, cmap="gray", vmin=0, vmax=1)
    else:
        plt.imshow(inter_img, vmin=0, vmax=1)
    plt.show()


def rgb2yiq(imRGB):
    """
    Transform an RGB image into the YIQ color space
    :param imRGB: height X width X 3 np.float64 matrix in the [0,1] range
    :return: the image in the YIQ space
    """
    return imRGB @ RGB_YIQ_TRANSFORMATION_MATRIX.T


def yiq2rgb(imYIQ):
    """
    Transform a YIQ image into the RGB color space
    :param imYIQ: height X width X 3 np.float64 matrix in the [0,1] range for
        the Y channel and in the range of [-1,1] for the I,Q channels
    :return: the image in the RGB space
    """
    return imYIQ @ YIQ_RGB_TRANSFORMATION_MATRIX.T


def histogram_equalize(im_orig):
    """
    Performs optimal equalization of a given greyscale or RGB image
    :param im_orig: Input float64 [0,1] image
    :return:  im_eq - is the equalized output image
              hist_orig - the image's original histogram before any changes
              hist_eq = equalized image's histogram after changes
    """
    im_yiq = np.zeros_like(im_orig)
    if len(np.shape(im_orig)) == 3:
        im_yiq = rgb2yiq(im_orig)
        y_channel = im_yiq[:, :, 0]
    else:
        y_channel = im_orig
    hist_orig, im_eq = equalize_helper(y_channel)
    if len(np.shape(im_orig)) == 3:
        im_yiq[:, :, 0] = im_eq
        im_eq = yiq2rgb(im_yiq)
    hist_eq = np.histogram(im_eq, bins=256, range=[0, 1])[0]
    return im_eq, hist_orig, hist_eq


def equalize_helper(y_channel):
    """
    Helper function for equalization process
    :param y_channel: the channel to preform equalization on
    :return:  im_eq - is the equalized output image
              hist_orig - the image's original histogram before any changes
    """
    norm_img = np.round((y_channel * 255).astype(np.uint8))
    hist_orig = np.histogram(norm_img, bins=256, range=[0, 256])[0]
    cumulative_sum = np.cumsum(hist_orig)
    c_m = min(cumulative_sum[cumulative_sum > 0])  # TODO Change
    T = np.round(((cumulative_sum - c_m) / (cumulative_sum[-1] - c_m)) * 255)
    im_eq = (T[norm_img]) / 255
    return hist_orig, im_eq


def compute_q_i(z, norm_hist, i):
    """
    Helper function for computin q_i
    :return:  corrct q_i
    """
    q_i = 0
    g = np.floor(z[i]) + 1
    g = g.astype(np.uint16)
    if i < (len(z) - 1):
        iter = np.floor(z[i + 1]) + 1 - np.floor(z[i])
    else:
        iter = np.floor(z[i] + 1) - np.floor(z[i])
    iter = int(iter)
    numerator = sum([norm_hist[g + j - 1] * (g + j - 1) for j in
                     range(iter)])
    denominator = sum([norm_hist[g + j - 1] for j in range(iter)])
    if denominator != 0:
        q_i = numerator / denominator
    return q_i


def compute_error(z, q, norm_hist, n_quants, i):
    """
    Helper function for computing the error
    :return: error for specific z_i - z_i+1 interval
    """
    start_idx = (np.floor(z[i]) + 1).astype(np.uint8)
    if i == n_quants - 1:
        end_idx = (len(norm_hist) - 1)
    else:
        end_idx = (np.floor(z[i + 1]) + 1).astype(np.uint8)
    g = np.arange(start_idx, end_idx + 1)
    err = np.sum(
        (norm_hist[start_idx:end_idx] * (q[i] - g[:(len(g) - 1)])**2))
    return err


def quantize(im_orig, n_quant, n_iter):
    """
    Performs optimal quantization of a given greyscale or RGB image
    :param im_orig: Input float64 [0,1] image
    :param n_quant: Number of intensities im_quant image will have
    :param n_iter: Maximum number of iterations of the optimization
    :return:  im_quant - is the quantized output image
              error - is an array with shape (n_iter,) (or less) of
                the total intensities error for each iteration of the
                quantization procedure
    """
    yiq_img = im_orig
    if len(np.shape(im_orig)) == 3:
        yiq_img = rgb2yiq(im_orig)
        y_channel = yiq_img[:, :, 0]
    else:
        y_channel = im_orig
    norm_img = np.round((y_channel * 255).astype(np.uint8))
    error = []
    norm_hist = np.histogram(norm_img, bins=256, range=[0, 256])[0]
    cum_sum = np.cumsum(norm_hist)
    img_width = np.shape(yiq_img)[0]
    img_length = np.shape(yiq_img)[1]
    z = np.zeros(n_quant + 1)
    z = z.astype(np.int16)
    init_divider_quant = (1 / n_quant) * img_width * img_length
    i = 1
    tmp = cum_sum
    while i < n_quant:
        z[i] = np.where(tmp[z[i]:] >= init_divider_quant)[0][0]
        tmp = np.subtract(tmp, np.round(init_divider_quant))
        i += 1
    z[n_quant] = 255
    q = np.zeros(n_quant + 1)
    k = len(z)
    q = quantize_helper(error, k, n_iter, norm_hist, q, z)
    new_hist = np.zeros_like(norm_hist)
    for i in range(k - 1):
        new_hist[z[i]:z[i + 1]] = q[i]
    new_hist[-1] = q[k - 2]
    im_quant = new_hist[norm_img.astype(np.uint8)]
    im_quant = im_quant / 255
    if len(np.shape(im_orig)) == 3:
        yiq_img[:, :, 0] = im_quant
        im_quant = yiq2rgb(yiq_img)
    return [im_quant, error]


def quantize_helper(error, k, n_iter, norm_hist, q, z):
    """
    Helper function to perform quantization process
    :param error: Input float64 [0,1] image
    :param k: Input float64 [0,1] image
    :param n_iter: Maximum number of iterations of the optimization
    :param n_quant: Number of intensities im_quant image will have
    :param norm_hist: normlized histogram of the image
    :param q: q values table to map to actual image
    :param z: z values table to decide the indices which will limit the
    intensities' borders
    :return:  q values table to map to actual image, represents the intesities
    """
    err = 0
    for i in range(n_iter):
        for j in range(k - 1):
            q[j] = compute_q_i(z, norm_hist, j)
        for m in range(k - 1):
            if m == 0:
                z[m] = 0
            else:
                z_i = ((q[m] + q[m - 1]) / 2)
                z_i = np.round(z_i).astype(int)
                z[m] = z_i
        for n in range(k - 1):
            err += compute_error(z, q, norm_hist, n_iter, n)
        error.append(err)
        q = q.astype(np.uint16)
        if i != 0:
            if error[i] - error[i - 1] <= 10000 or error[i] - error[i - 1] <= \
                    10000:
                break
        err = 0
    return q


def quantize_rgb(im_orig, n_quant):  # Bonus - optional
    """
    Performs optimal quantization of a given greyscale or RGB image
    :param im_orig: Input RGB image of type float64 in the range [0,1]
    :param n_quant: Number of intensities im_quant image will have
    :return:  im_quant - the quantized output image
    """
    pass


