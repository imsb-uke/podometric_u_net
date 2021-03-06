import numpy as np
from skimage.transform import AffineTransform
from scipy import ndimage as ndi

from skimage.transform._warps_cy import _warp_fast
from skimage._shared.utils import warn, safe_as_int, get_bound_method_class
from skimage.transform._geometric import (SimilarityTransform, AffineTransform,
                         ProjectiveTransform, _to_ndimage_mode)


HOMOGRAPHY_TRANSFORMS = (
    SimilarityTransform,
    AffineTransform,
    ProjectiveTransform
)


def printGreen(*string):
    for element in string:
        print("\033[92m {}\033[00m".format(element))


def rescale_array(array, scale, order=1, mode='constant', cval=0, clip=True, multichannel=True, anti_aliasing=True,
                  anti_aliasing_sigma=None, verbose=False):
    """
       Modified from https://github.com/scikit-image/scikit-image/blob/master/skimage/transform/_warps.py#L19

       Scale image by a certain factor.
       Performs interpolation to up-scale or down-scale N-dimensional images.
       Note that anti-aliasing should be enabled when down-sizing images to avoid
       aliasing artifacts. For down-sampling with an integer factor also see
       `skimage.transform.downscale_local_mean`.
       Parameters
       ----------
       array : ndarray
           Input array.
       scale : {float, tuple of floats}
           Scale factors. Separate scale factors can be defined as
           `(rows, cols[, ...][, dim])`.
       Returns
       -------
       scaled : ndarray
           Scaled version of the input.
       Other parameters
       ----------------
       order : int, optional
           The order of the spline interpolation, default is 1. The order has to
           be in the range 0-5. See `skimage.transform.warp` for detail.
       mode : {'constant', 'edge', 'symmetric', 'reflect', 'wrap'}, optional
           Points outside the boundaries of the input are filled according
           to the given mode.  Modes match the behaviour of `numpy.pad`.
       cval : float, optional
           Used in conjunction with mode 'constant', the value outside
           the image boundaries.
       clip : bool, optional
           Whether to clip the output to the range of values of the input image.
           This is enabled by default, since higher order interpolation may
           produce values outside the given input range.
       multichannel : bool, optional
           Whether the last axis of the image is to be interpreted as multiple
           channels or another spatial dimension.
       anti_aliasing : bool, optional
           Whether to apply a Gaussian filter to smooth the image prior to
           down-scaling. It is crucial to filter when down-sampling the image to
           avoid aliasing artifacts.
       anti_aliasing_sigma : {float, tuple of floats}, optional
           Standard deviation for Gaussian filtering to avoid aliasing artifacts.
           By default, this value is chosen as (1 - s) / 2 where s is the
           down-scaling factor.
       Notes
       -----
       Modes 'reflect' and 'symmetric' are similar, but differ in whether the edge
       pixels are duplicated during the reflection.  As an example, if an array
       has values [0, 1, 2] and was padded to the right by four values using
       symmetric, the result would be [0, 1, 2, 2, 1, 0, 0], while for reflect it
       would be [0, 1, 2, 1, 0, 1, 2].
       Examples
       --------
       >>> array = data.xyz()
       >>> rescale(array, 0.1).shape
       (51, 51)
       >>> rescale(array, 0.5).shape
       (256, 256)
       """
    if verbose:
        printGreen("DEBUG: rescale_array, weight array shape/min/max", array.shape, array.min(), array.max())

    scale = np.atleast_1d(scale)
    if len(scale) > 1:
        if ((not multichannel and len(scale) != array.ndim) or
                (multichannel and len(scale) != array.ndim - 1)):
            raise ValueError("Supply a single scale, or one value per spatial "
                             "axis")
        if multichannel:
            scale = np.concatenate((scale, [1]))
    orig_shape = np.asarray(array.shape)
    output_shape = np.round(scale * orig_shape)
    if multichannel:  # don't scale channel dimension
        output_shape[-1] = orig_shape[-1]

    return resize(array, output_shape, order=order, mode=mode, cval=cval,
                  clip=clip,
                  anti_aliasing=anti_aliasing,
                  anti_aliasing_sigma=anti_aliasing_sigma,
                  verbose=verbose)


def resize(array, output_shape, order=1, mode='reflect', cval=0, clip=True,
           anti_aliasing=True, anti_aliasing_sigma=None, verbose=False):
    """Resize image to match a certain size.
    Performs interpolation to up-size or down-size N-dimensional images. Note
    that anti-aliasing should be enabled when down-sizing images to avoid
    aliasing artifacts. For down-sampling with an integer factor also see
    `skimage.transform.downscale_local_mean`.
    Parameters
    ----------
    image : ndarray
        Input image.
    output_shape : tuple or ndarray
        Size of the generated output image `(rows, cols[, ...][, dim])`. If
        `dim` is not provided, the number of channels is preserved. In case the
        number of input channels does not equal the number of output channels a
        n-dimensional interpolation is applied.
    Returns
    -------
    resized : ndarray
        Resized version of the input.
    Other parameters
    ----------------
    order : int, optional
        The order of the spline interpolation, default is 1. The order has to
        be in the range 0-5. See `skimage.transform.warp` for detail.
    mode : {'constant', 'edge', 'symmetric', 'reflect', 'wrap'}, optional
        Points outside the boundaries of the input are filled according
        to the given mode.  Modes match the behaviour of `numpy.pad`.
    cval : float, optional
        Used in conjunction with mode 'constant', the value outside
        the image boundaries.
    clip : bool, optional
        Whether to clip the output to the range of values of the input image.
        This is enabled by default, since higher order interpolation may
        produce values outside the given input range.
    anti_aliasing : bool, optional
        Whether to apply a Gaussian filter to smooth the image prior to
        down-scaling. It is crucial to filter when down-sampling the image to
        avoid aliasing artifacts.
    anti_aliasing_sigma : {float, tuple of floats}, optional
        Standard deviation for Gaussian filtering to avoid aliasing artifacts.
        By default, this value is chosen as (s - 1) / 2 where s is the
        down-scaling factor, where s > 1. For the up-size case, s < 1, no
        anti-aliasing is performed prior to rescaling.
    Notes
    -----
    Modes 'reflect' and 'symmetric' are similar, but differ in whether the edge
    pixels are duplicated during the reflection.  As an example, if an array
    has values [0, 1, 2] and was padded to the right by four values using
    symmetric, the result would be [0, 1, 2, 2, 1, 0, 0], while for reflect it
    would be [0, 1, 2, 1, 0, 1, 2].
    Examples
    --------
    >>> array = data.xyz()
    >>> resize(array, (100, 100)).shape
    (100, 100)
    """
    if verbose:
        printGreen("DEBUG: resize_array, weight array shape/min/max", array.shape, array.min(), array.max())

    output_shape = tuple(output_shape)
    output_ndim = len(output_shape)
    input_shape = array.shape
    if output_ndim > array.ndim:
        # append dimensions to input_shape
        input_shape = input_shape + (1, ) * (output_ndim - array.ndim)
        array = np.reshape(array, input_shape)
    elif output_ndim == array.ndim - 1:
        # multichannel case: append shape of last axis
        output_shape = output_shape + (array.shape[-1], )
    elif output_ndim < array.ndim - 1:
        raise ValueError("len(output_shape) cannot be smaller than the array "
                         "dimensions")

    factors = (np.asarray(input_shape, dtype=float) /
               np.asarray(output_shape, dtype=float))

    if anti_aliasing:
        if anti_aliasing_sigma is None:
            anti_aliasing_sigma = np.maximum(0, (factors - 1) / 2)
        else:
            anti_aliasing_sigma = \
                np.atleast_1d(anti_aliasing_sigma) * np.ones_like(factors)
            if np.any(anti_aliasing_sigma < 0):
                raise ValueError("Anti-aliasing standard deviation must be "
                                 "greater than or equal to zero")
            elif np.any((anti_aliasing_sigma > 0) & (factors <= 1)):
                warn("Anti-aliasing standard deviation greater than zero but "
                     "not down-sampling along all axes")

        # Translate modes used by np.pad to those used by ndi.gaussian_filter
        np_pad_to_ndimage = {
            'constant': 'constant',
            'edge': 'nearest',
            'symmetric': 'reflect',
            'reflect': 'mirror',
            'wrap': 'wrap'
        }
        try:
            ndi_mode = np_pad_to_ndimage[mode]
        except KeyError:
            raise ValueError("Unknown mode, or cannot translate mode. The "
                             "mode should be one of 'constant', 'edge', "
                             "'symmetric', 'reflect', or 'wrap'. See the "
                             "documentation of numpy.pad for more info.")

        array = ndi.gaussian_filter(array, anti_aliasing_sigma,
                                    cval=cval, mode=ndi_mode)

    # 2-dimensional interpolation
    if len(output_shape) == 2 or (len(output_shape) == 3 and
                                  output_shape[2] == input_shape[2]):
        rows = output_shape[0]
        cols = output_shape[1]
        input_rows = input_shape[0]
        input_cols = input_shape[1]
        if rows == 1 and cols == 1:
            tform = AffineTransform(translation=(input_cols / 2.0 - 0.5,
                                                 input_rows / 2.0 - 0.5))
        else:
            # 3 control points necessary to estimate exact AffineTransform
            src_corners = np.array([[1, 1], [1, rows], [cols, rows]]) - 1
            dst_corners = np.zeros(src_corners.shape, dtype=np.double)
            # take into account that 0th pixel is at position (0.5, 0.5)
            dst_corners[:, 0] = factors[1] * (src_corners[:, 0] + 0.5) - 0.5
            dst_corners[:, 1] = factors[0] * (src_corners[:, 1] + 0.5) - 0.5

            tform = AffineTransform()
            tform.estimate(src_corners, dst_corners)

        # Make sure the transform is exactly metric, to ensure fast warping.
        tform.params[2] = (0, 0, 1)
        tform.params[0, 1] = 0
        tform.params[1, 0] = 0

        out = warp(array, tform, output_shape=output_shape, order=order,
                   mode=mode, cval=cval, clip=clip)

    else:  # n-dimensional interpolation
        coord_arrays = [factors[i] * (np.arange(d) + 0.5) - 0.5
                        for i, d in enumerate(output_shape)]

        coord_map = np.array(np.meshgrid(*coord_arrays,
                                         sparse=False,
                                         indexing='ij'))

        # Do not convert array
        #array = convert_to_float(array, preserve_range)

        ndi_mode = _to_ndimage_mode(mode)
        out = ndi.map_coordinates(array, coord_map, order=order,
                                  mode=ndi_mode, cval=cval)

        _clip_warp_output(array, out, order, mode, cval, clip)

    if verbose:
        printGreen("DEBUG: resize DOWN, weight array shape/min/max", out.shape, out.min(), out.max())

    return out


def warp(image, inverse_map, map_args={}, output_shape=None, order=1,
         mode='constant', cval=0., clip=True):
    """Warp an image according to a given coordinate transformation.
    Parameters
    ----------
    image : ndarray
        Input image.
    inverse_map : transformation object, callable ``cr = f(cr, **kwargs)``, or ndarray
        Inverse coordinate map, which transforms coordinates in the output
        images into their corresponding coordinates in the input image.
        There are a number of different options to define this map, depending
        on the dimensionality of the input image. A 2-D image can have 2
        dimensions for gray-scale images, or 3 dimensions with color
        information.
         - For 2-D images, you can directly pass a transformation object,
           e.g. `skimage.transform.SimilarityTransform`, or its inverse.
         - For 2-D images, you can pass a ``(3, 3)`` homogeneous
           transformation matrix, e.g.
           `skimage.transform.SimilarityTransform.params`.
         - For 2-D images, a function that transforms a ``(M, 2)`` array of
           ``(col, row)`` coordinates in the output image to their
           corresponding coordinates in the input image. Extra parameters to
           the function can be specified through `map_args`.
         - For N-D images, you can directly pass an array of coordinates.
           The first dimension specifies the coordinates in the input image,
           while the subsequent dimensions determine the position in the
           output image. E.g. in case of 2-D images, you need to pass an array
           of shape ``(2, rows, cols)``, where `rows` and `cols` determine the
           shape of the output image, and the first dimension contains the
           ``(row, col)`` coordinate in the input image.
           See `scipy.ndimage.map_coordinates` for further documentation.
        Note, that a ``(3, 3)`` matrix is interpreted as a homogeneous
        transformation matrix, so you cannot interpolate values from a 3-D
        input, if the output is of shape ``(3,)``.
        See example section for usage.
    map_args : dict, optional
        Keyword arguments passed to `inverse_map`.
    output_shape : tuple (rows, cols), optional
        Shape of the output image generated. By default the shape of the input
        image is preserved.  Note that, even for multi-band images, only rows
        and columns need to be specified.
    order : int, optional
        The order of interpolation. The order has to be in the range 0-5:
         - 0: Nearest-neighbor
         - 1: Bi-linear (default)
         - 2: Bi-quadratic
         - 3: Bi-cubic
         - 4: Bi-quartic
         - 5: Bi-quintic
    mode : {'constant', 'edge', 'symmetric', 'reflect', 'wrap'}, optional
        Points outside the boundaries of the input are filled according
        to the given mode.  Modes match the behaviour of `numpy.pad`.
    cval : float, optional
        Used in conjunction with mode 'constant', the value outside
        the image boundaries.
    clip : bool, optional
        Whether to clip the output to the range of values of the input image.
        This is enabled by default, since higher order interpolation may
        produce values outside the given input range.
    preserve_range : bool, optional
        Whether to keep the original range of values. Otherwise, the input
        image is converted according to the conventions of `img_as_float`.
        Also see
        https://scikit-image.org/docs/dev/user_guide/data_types.html
    Returns
    -------
    warped : double ndarray
        The warped input image.
    Notes
    -----
    - The input image is converted to a `double` image.
    - In case of a `SimilarityTransform`, `AffineTransform` and
      `ProjectiveTransform` and `order` in [0, 3] this function uses the
      underlying transformation matrix to warp the image with a much faster
      routine.
    Examples
    --------
    >>> from skimage.transform import warp
    >>> from skimage import data
    >>> image = data.camera()
    The following image warps are all equal but differ substantially in
    execution time. The image is shifted to the bottom.
    Use a geometric transform to warp an image (fast):
    >>> from skimage.transform import SimilarityTransform
    >>> tform = SimilarityTransform(translation=(0, -10))
    >>> warped = warp(image, tform)
    Use a callable (slow):
    >>> def shift_down(xy):
    ...     xy[:, 1] -= 10
    ...     return xy
    >>> warped = warp(image, shift_down)
    Use a transformation matrix to warp an image (fast):
    >>> matrix = np.array([[1, 0, 0], [0, 1, -10], [0, 0, 1]])
    >>> warped = warp(image, matrix)
    >>> from skimage.transform import ProjectiveTransform
    >>> warped = warp(image, ProjectiveTransform(matrix=matrix))
    You can also use the inverse of a geometric transformation (fast):
    >>> warped = warp(image, tform.inverse)
    For N-D images you can pass a coordinate array, that specifies the
    coordinates in the input image for every element in the output image. E.g.
    if you want to rescale a 3-D cube, you can do:
    >>> cube_shape = np.array([30, 30, 30])
    >>> cube = np.random.rand(*cube_shape)
    Setup the coordinate array, that defines the scaling:
    >>> scale = 0.1
    >>> output_shape = (scale * cube_shape).astype(int)
    >>> coords0, coords1, coords2 = np.mgrid[:output_shape[0],
    ...                    :output_shape[1], :output_shape[2]]
    >>> coords = np.array([coords0, coords1, coords2])
    Assume that the cube contains spatial data, where the first array element
    center is at coordinate (0.5, 0.5, 0.5) in real space, i.e. we have to
    account for this extra offset when scaling the image:
    >>> coords = (coords + 0.5) / scale - 0.5
    >>> warped = warp(cube, coords)
    """

    if image.size == 0:
        raise ValueError("Cannot warp empty image with dimensions", image.shape)

    # Do not convert the image because it is an array here!
    #image = convert_to_float(image, preserve_range)

    input_shape = np.array(image.shape)

    if output_shape is None:
        output_shape = input_shape
    else:
        output_shape = safe_as_int(output_shape)

    warped = None

    if order == 2:
        # When fixing this issue, make sure to fix the branches further
        # below in this function
        warn("Bi-quadratic interpolation behavior has changed due "
             "to a bug in the implementation of scikit-image. "
             "The new version now serves as a wrapper "
             "around SciPy's interpolation functions, which itself "
             "is not verified to be a correct implementation. Until "
             "skimage's implementation is fixed, we recommend "
             "to use bi-linear or bi-cubic interpolation instead.")

    if order in (0, 1, 3) and not map_args:
        # use fast Cython version for specific interpolation orders and input

        matrix = None

        if isinstance(inverse_map, np.ndarray) and inverse_map.shape == (3, 3):
            # inverse_map is a transformation matrix as numpy array
            matrix = inverse_map

        elif isinstance(inverse_map, HOMOGRAPHY_TRANSFORMS):
            # inverse_map is a homography
            matrix = inverse_map.params

        elif (hasattr(inverse_map, '__name__') and
              inverse_map.__name__ == 'inverse' and
              get_bound_method_class(inverse_map) in HOMOGRAPHY_TRANSFORMS):
            # inverse_map is the inverse of a homography
            matrix = np.linalg.inv(inverse_map.__self__.params)

        if matrix is not None:
            matrix = matrix.astype(np.double)
            if image.ndim == 2:
                warped = _warp_fast(image, matrix,
                                    output_shape=output_shape,
                                    order=order, mode=mode, cval=cval)
            elif image.ndim == 3:
                dims = []
                for dim in range(image.shape[2]):
                    dims.append(_warp_fast(image[..., dim], matrix,
                                           output_shape=output_shape,
                                           order=order, mode=mode, cval=cval))
                warped = np.dstack(dims)

    if warped is None:
        # use ndi.map_coordinates

        if (isinstance(inverse_map, np.ndarray) and
                inverse_map.shape == (3, 3)):
            # inverse_map is a transformation matrix as numpy array,
            # this is only used for order >= 4.
            inverse_map = ProjectiveTransform(matrix=inverse_map)

        if isinstance(inverse_map, np.ndarray):
            # inverse_map is directly given as coordinates
            coords = inverse_map
        else:
            # inverse_map is given as function, that transforms (N, 2)
            # destination coordinates to their corresponding source
            # coordinates. This is only supported for 2(+1)-D images.

            if image.ndim < 2 or image.ndim > 3:
                raise ValueError("Only 2-D images (grayscale or color) are "
                                 "supported, when providing a callable "
                                 "`inverse_map`.")

            def coord_map(*args):
                return inverse_map(*args, **map_args)

            if len(input_shape) == 3 and len(output_shape) == 2:
                # Input image is 2D and has color channel, but output_shape is
                # given for 2-D images. Automatically add the color channel
                # dimensionality.
                output_shape = (output_shape[0], output_shape[1],
                                input_shape[2])

            coords = warp_coords(coord_map, output_shape)

        # Pre-filtering not necessary for order 0, 1 interpolation
        prefilter = order > 1

        ndi_mode = _to_ndimage_mode(mode)
        warped = ndi.map_coordinates(image, coords, prefilter=prefilter,
                                     mode=ndi_mode, order=order, cval=cval)

    _clip_warp_output(image, warped, order, mode, cval, clip)

    return warped


def _clip_warp_output(input_image, output_image, order, mode, cval, clip):
    """Clip output image to range of values of input image.
    Note that this function modifies the values of `output_image` in-place
    and it is only modified if ``clip=True``.
    Parameters
    ----------
    input_image : ndarray
        Input image.
    output_image : ndarray
        Output image, which is modified in-place.
    Other parameters
    ----------------
    order : int, optional
        The order of the spline interpolation, default is 1. The order has to
        be in the range 0-5. See `skimage.transform.warp` for detail.
    mode : {'constant', 'edge', 'symmetric', 'reflect', 'wrap'}, optional
        Points outside the boundaries of the input are filled according
        to the given mode.  Modes match the behaviour of `numpy.pad`.
    cval : float, optional
        Used in conjunction with mode 'constant', the value outside
        the image boundaries.
    clip : bool, optional
        Whether to clip the output to the range of values of the input image.
        This is enabled by default, since higher order interpolation may
        produce values outside the given input range.
    """
    if clip and order != 0:
        min_val = input_image.min()
        max_val = input_image.max()

        preserve_cval = (mode == 'constant' and not
                         (min_val <= cval <= max_val))

        if preserve_cval:
            cval_mask = output_image == cval

        np.clip(output_image, min_val, max_val, out=output_image)

        if preserve_cval:
            output_image[cval_mask] = cval


def _stackcopy(a, b):
    """Copy b into each color layer of a, such that::
      a[:,:,0] = a[:,:,1] = ... = b
    Parameters
    ----------
    a : (M, N) or (M, N, P) ndarray
        Target array.
    b : (M, N)
        Source array.
    Notes
    -----
    Color images are stored as an ``(M, N, 3)`` or ``(M, N, 4)`` arrays.
    """
    if a.ndim == 3:
        a[:] = b[:, :, np.newaxis]
    else:
        a[:] = b


def warp_coords(coord_map, shape, dtype=np.float64):
    """Build the source coordinates for the output of a 2-D image warp.
    Parameters
    ----------
    coord_map : callable like GeometricTransform.inverse
        Return input coordinates for given output coordinates.
        Coordinates are in the shape (P, 2), where P is the number
        of coordinates and each element is a ``(row, col)`` pair.
    shape : tuple
        Shape of output image ``(rows, cols[, bands])``.
    dtype : np.dtype or string
        dtype for return value (sane choices: float32 or float64).
    Returns
    -------
    coords : (ndim, rows, cols[, bands]) array of dtype `dtype`
            Coordinates for `scipy.ndimage.map_coordinates`, that will yield
            an image of shape (orows, ocols, bands) by drawing from source
            points according to the `coord_transform_fn`.
    Notes
    -----
    This is a lower-level routine that produces the source coordinates for 2-D
    images used by `warp()`.
    It is provided separately from `warp` to give additional flexibility to
    users who would like, for example, to re-use a particular coordinate
    mapping, to use specific dtypes at various points along the the
    image-warping process, or to implement different post-processing logic
    than `warp` performs after the call to `ndi.map_coordinates`.
    Examples
    --------
    Produce a coordinate map that shifts an image up and to the right:
    >>> from skimage import data
    >>> from scipy.ndimage import map_coordinates
    >>>
    >>> def shift_up10_left20(xy):
    ...     return xy - np.array([-20, 10])[None, :]
    >>>
    >>> image = data.astronaut().astype(np.float32)
    >>> coords = warp_coords(shift_up10_left20, image.shape)
    >>> warped_image = map_coordinates(image, coords)
    """
    shape = safe_as_int(shape)
    rows, cols = shape[0], shape[1]
    coords_shape = [len(shape), rows, cols]
    if len(shape) == 3:
        coords_shape.append(shape[2])
    coords = np.empty(coords_shape, dtype=dtype)

    # Reshape grid coordinates into a (P, 2) array of (row, col) pairs
    tf_coords = np.indices((cols, rows), dtype=dtype).reshape(2, -1).T

    # Map each (row, col) pair to the source image according to
    # the user-provided mapping
    tf_coords = coord_map(tf_coords)

    # Reshape back to a (2, M, N) coordinate grid
    tf_coords = tf_coords.T.reshape((-1, cols, rows)).swapaxes(1, 2)

    # Place the y-coordinate mapping
    _stackcopy(coords[1, ...], tf_coords[0, ...])

    # Place the x-coordinate mapping
    _stackcopy(coords[0, ...], tf_coords[1, ...])

    if len(shape) == 3:
        coords[2, ...] = range(shape[2])

    return coords