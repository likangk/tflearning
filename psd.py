epsilon = 1e-8
def psd_trans(img):
    img = img.numpy()
    complex_data = np.complex(img, np.zeros_like(img))#转换为tf用的complex格式
    sps_data = np.fft.fft2(complex_data)#傅里叶变换
    sps_data = np.fft.fftshift(sps_data)
    sps_data = epsilon + sps_data
    magnitude_spectrum = 20*np.log(np.abs(sps_data))
    psd1D = azimuthalAverage(magnitude_spectrum)
    psd1D = (psd1D-np.min(psd1D))/(np.max(psd1D)-np.min(psd1D))
    return psd1D
    #psd1D = torch.from_numpy(psd1D).float()

def azimuthalAverage(image, center=None):
    """
    Calculate the azimuthally averaged radial profile.
    image - The 2D image
    center - The [x,y] pixel coordinates used as the center. The default is 
             None, which then uses the center of the image (including 
             fracitonal pixels).
    
    """
    # Calculate the indices from the image
    y, x = np.indices(image.shape)

    if not center:
        center = np.array([(x.max()-x.min())/2.0, (y.max()-y.min())/2.0])

    r = np.hypot(x - center[0], y - center[1])

    # Get sorted radii
    ind = np.argsort(r.flat)
    r_sorted = r.flat[ind]
    i_sorted = image.flat[ind]

    # Get the integer part of the radii (bin size = 1)
    r_int = r_sorted.astype(int)

    # Find all pixels that fall within each radial bin.
    deltar = r_int[1:] - r_int[:-1]  # Assumes all radii represented
    rind = np.where(deltar)[0]       # location of changed radius
    nr = rind[1:] - rind[:-1]        # number of radius bin
    
    # Cumulative sum to figure out sums for each radius bin
    csim = np.cumsum(i_sorted, dtype=float)
    tbin = csim[rind[1:]] - csim[rind[:-1]]

    radial_prof = tbin / nr

    return radial_prof


def generator_loss(disc_generated_output, gen_output, target):
  gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)

  # mean absolute error
  l1_loss = tf.reduce_mean(tf.abs(target - gen_output))
  #psd_loss
  psd_loss = loss_object(psd_trans(target),psd_trans(gen_output))

  total_gen_loss = gan_loss + (LAMBDA * l1_loss) + psd_loss * gan_loss * 1e-5

  return total_gen_loss, gan_loss, l1_loss