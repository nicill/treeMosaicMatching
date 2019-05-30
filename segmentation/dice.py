import sys, getopt
import numpy as np
from PIL import Image
from PIL import ImageOps
import warnings

def dice(im1, im2, empty_score=1.0):
    """
    Computes the Dice coefficient, a measure of set similarity.
    Parameters
    ----------
    im1 : array-like, bool
        Any array of arbitrary size. If not boolean, will be converted.
    im2 : array-like, bool
        Any other array of identical size. If not boolean, will be converted.
    Returns
    -------
    dice : float
        Dice coefficient as a float on range [0,1].
        Maximum similarity = 1
        No similarity = 0
        Both are empty (sum eq to zero) = empty_score

    Notes
    -----
    The order of inputs for `dice` is irrelevant. The result will be
    identical if `im1` and `im2` are switched.
    """
    im1 = np.asarray(im1).astype(np.bool)
    im2 = np.asarray(im2).astype(np.bool)

    #print("im1: " + repr(im1.shape))
    #print("im2: " + repr(im2.shape))

    if im1.shape != im2.shape:
        print("im1: " + repr(im1.shape))
        print("im1: " + repr(im1.shape))
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    im_sum = im1.sum() + im2.sum()
    if im_sum == 0:
        return empty_score

    # Compute Dice coefficient
    intersection = np.logical_and(im1, im2)

    return 2. * intersection.sum() / im_sum

print ("Dice coefficient between two boolean images")
print ("-------------------------------------------")



def main(argv):
    warnings.simplefilter('ignore', Image.DecompressionBombWarning)

    im1 = Image.open(sys.argv[1])
    im2 = Image.open(sys.argv[2])

    # Error control if only one parameter for inverting image is given.
    if len(sys.argv) == 4:
        print("Invert options must be set for both images")
        sys.exit()

    # Inverting input images if is wanted.
    if len(sys.argv) > 3:
        if int(sys.argv[3]) == 1:
            im1 = ImageOps.invert(im1)

        if int(sys.argv[4]) == 1:
            im2 = ImageOps.invert(im2)

    # Image resize for squared images.
    size = 300, 300
    # im1.thumbnail(size, Image.ANTIALIAS)
    # im1.show()
    # im2.thumbnail(size, Image.ANTIALIAS)
    # im2.show()

    # Converting to b/w.
    gray1 = im1.convert('L')
    im1 = gray1.point(lambda x: 0 if x < 128 else 255, '1')
    gray2 = im2.convert('L')
    im2 = gray2.point(lambda x: 0 if x < 128 else 255, '1')

    # Dice coeficinet computation
    res = dice(im1, im2)

    print(res)

if __name__ == "__main__":

    main(sys.argv[1:])
