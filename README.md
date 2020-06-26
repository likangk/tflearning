# tflearning
```
def psd_trans(img):
    fft = np.fft.fft2(img)
    fshift = np.fft.fftshift(fft)
    fshift += epsilon
    image = 20*np.log(np.abs(fshift))
    y, x = np.indices(image.shape)
    center = None
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
for test_input, tar in test_dataset.take(1):
    prediction = generator(test_input, training=True).numpy()[0,:,:,0]
    tar = tar.numpy()[0,:,:,0]
    groudtruth = psd_trans(tar)
    output = psd_trans(prediction)
    s_len = range(len(output))
    plt.plot(s_len, groudtruth, marker='.', mec='r', mfc='w',label='groud truth')
    plt.plot(s_len, output, marker='.', ms=10,label='output')
    plt.legend()  # 让图例生效
 
    plt.margins(0)
    plt.subplots_adjust(bottom=0.10)
    plt.xlabel('the length') #X轴标签
    plt.ylabel("f1") #Y轴标签
