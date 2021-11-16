from typing import Callable, List

import numpy as np
import scipy.signal

WINDOW_CACHE = dict()


def _spline_window(window_size: int, power: int = 2) -> np.ndarray:
    """Generates a 1-dimensional spline of order 'power' (typically 2), in the designated
    window.
    https://www.wolframalpha.com/input/?i=y%3Dx**2,+y%3D-(x-2)**2+%2B2,+y%3D(x-4)**2,+from+y+%3D+0+to+2

    Args:
        window_size (int): size of the interested window
        power (int, optional): Order of the spline. Defaults to 2.

    Returns:
        np.ndarray: 1D spline
    """
    intersection = int(window_size / 4)
    wind_outer = (abs(2 * (scipy.signal.triang(window_size)))**power) / 2
    wind_outer[intersection:-intersection] = 0

    wind_inner = 1 - (abs(2 * (scipy.signal.triang(window_size) - 1))**power) / 2
    wind_inner[:intersection] = 0
    wind_inner[-intersection:] = 0

    wind = wind_inner + wind_outer
    wind = wind / np.average(wind)
    return wind


def _spline_2d(window_size: int, power: int = 2):
    """Makes a 1D window spline function, then combines it to return a 2D window function.
    The 2D window is useful to smoothly interpolate between patches.

    Args:
        window_size (int): size of the window (patch)
        power (int, optional): Which order for the spline. Defaults to 2.

    Returns:
        np.ndarray: numpy array containing a 2D spline function
    """
    # Memorization to avoid remaking it for every call
    # since the same window is needed multiple times
    global WINDOW_CACHE
    key = f"{window_size}_{power}"
    if key in WINDOW_CACHE:
        wind = WINDOW_CACHE[key]
    else:
        wind = _spline_window(window_size, power)
        wind = np.expand_dims(np.expand_dims(wind, 1), 1)  # SREENI: Changed from 3, 3, to 1, 1
        wind = wind * wind.transpose(1, 0, 2)
        WINDOW_CACHE[key] = wind
    return wind


def pad_image(image: np.ndarray, tile_size: int, subdivisions: int) -> np.ndarray:
    """Add borders to the given image for a "valid" border pattern according to "window_size" and "subdivisions".
    Image is expected as a numpy array with shape (width, height, channels).

    Args:
        image (np.ndarray): input image, 3D channels-last tensor
        tile_size (int): size of a single patch, useful to compute padding
        subdivisions (int): amount of overlap, useful for padding

    Returns:
        np.ndarray: same image, padded specularly by a certain amount in every direction
    """
    # compute the pad as (window - window/subdivisions)
    pad = int(round(tile_size * (1 - 1.0 / subdivisions)))
    # add pad pixels in height and width, nothing channel-wise of course
    borders = ((pad, pad), (pad, pad), (0, 0))
    return np.pad(image, pad_width=borders, mode='reflect')


def unpad_image(padded_image: np.ndarray, tile_size: int, subdivisions: int) -> np.ndarray:
    """Reverts changes made by 'pad_image'. The same padding is removed, so tile_size and subdivisions
    must be coherent.

    Args:
        padded_image (np.ndarray): image with padding still applied
        tile_size (int): size of a single patch
        subdivisions (int): subdivisions to compute overlap

    Returns:
        np.ndarray: image without padding, 2D channels-last tensor
    """
    # compute the same amount as before, window - window/subdivisions
    pad = int(round(tile_size * (1 - 1.0 / subdivisions)))
    # crop the image left, right, top and bottom
    result = padded_image[pad:-pad, pad:-pad, :]
    return result


def rotate_and_mirror(image: np.ndarray) -> List[np.ndarray]:
    """Duplicates an image with shape (h, w, channels) 8 times, in order
    to have all the possible rotations and mirrors of that image that fits the
    possible 90 degrees rotations. https://en.wikipedia.org/wiki/Dihedral_group

    Args:
        image (np.ndarray): input image, already padded.

    Returns:
        List[np.ndarray]: list of images, rotated and mirrored.
    """
    variants = []
    variants.append(np.array(image))
    variants.append(np.rot90(np.array(image), axes=(0, 1), k=1))
    variants.append(np.rot90(np.array(image), axes=(0, 1), k=2))
    variants.append(np.rot90(np.array(image), axes=(0, 1), k=3))
    image = np.array(image)[:, ::-1]
    variants.append(np.array(image))
    variants.append(np.rot90(np.array(image), axes=(0, 1), k=1))
    variants.append(np.rot90(np.array(image), axes=(0, 1), k=2))
    variants.append(np.rot90(np.array(image), axes=(0, 1), k=3))
    return variants


def undo_rotate_and_mirror(variants: List[np.ndarray]) -> np.ndarray:
    """Reverts the 8 duplications provided by rotate and mirror.
    This restores the transformed inputs to the original position, then averages them.

    Args:
        variants (List[np.ndarray]): D4 dihedral group of the same image

    Returns:
        np.ndarray: averaged result over the given input.
    """
    origs = []
    origs.append(np.array(variants[0]))
    origs.append(np.rot90(np.array(variants[1]), axes=(0, 1), k=3))
    origs.append(np.rot90(np.array(variants[2]), axes=(0, 1), k=2))
    origs.append(np.rot90(np.array(variants[3]), axes=(0, 1), k=1))
    origs.append(np.array(variants[4])[:, ::-1])
    origs.append(np.rot90(np.array(variants[5]), axes=(0, 1), k=3)[:, ::-1])
    origs.append(np.rot90(np.array(variants[6]), axes=(0, 1), k=2)[:, ::-1])
    origs.append(np.rot90(np.array(variants[7]), axes=(0, 1), k=1)[:, ::-1])
    return np.mean(origs, axis=0)


def windowed_generator(padded_image: np.ndarray, window_size: int, subdivisions: int, batch_size: int = None):
    """Generator that yield tiles grouped by batch size.

    Args:
        padded_image (np.ndarray): input image to be processed (already padded)
        window_size (int): size of a single patch
        subdivisions (int): subdivision count on each patch to compute the step
        batch_size (int, optional): amount of patches in each batch. Defaults to None.

    Yields:
        Tuple[List[tuple], np.ndarray]: list of coordinates and respective patches as single batch array
    """
    step = window_size // subdivisions
    width, height, _ = padded_image.shape
    batch_size = batch_size or 1

    batch = []
    coords = []
    # step with fixed window on the image to build up the arrays
    for x in range(0, width - window_size + 1, step):
        for y in range(0, height - window_size + 1, step):
            coords.append((x, y))
            batch.append(padded_image[x:x + window_size, y:y + window_size])
            # yield the batch once full and restore lists right after
            if len(batch) == batch_size:
                yield coords, np.array(batch)
                coords.clear()
                batch.clear()
    # handle last (possibly unfinished) batch
    if len(batch) > 0:
        yield coords, np.array(batch)


def reconstruct(canvas: np.ndarray, tile_size: int, coords: List[tuple], predictions: np.ndarray) -> np.ndarray:
    """Helper function that iterates the result batch onto the given canvas to reconstruct
    the final result batch after batch.

    Args:
        canvas (np.ndarray): container for the final image.
        tile_size (int): size of a single patch.
        coords (List[tuple]): list of pixel coordinates corresponding to the batch items
        predictions (np.ndarray): array containing patch predictions, shape (batch, tile_size, tile_size, num_classes)

    Returns:
        np.ndarray: the updated canvas, shape (padded_w, padded_h, num_classes)
    """
    for (x, y), patch in zip(coords, predictions):
        canvas[x:x + tile_size, y:y + tile_size] += patch
    return canvas


def predict_smooth_windowing(image: np.ndarray,
                             tile_size: int,
                             subdivisions: int,
                             num_classes: int,
                             prediction_fn: Callable,
                             batch_size: int = None,
                             mirrored: bool = False) -> np.ndarray:
    """Allows to predict a large image in one go, dividing it in squared, fixed-size tiles and smoothly
    interpolating over them to produce a single, coherent output with the same dimensions.

    Args:
        image (np.ndarray): input image, expected a 3D vector
        tile_size (int): size of each squared tile
        subdivisions (int): number of subdivisions over the single tile for overlaps
        num_classes (int): number of output classes, required to rebuild
        prediction_fn (Callable): callback that takes the input batch and returns an output tensor
        batch_size (int, optional): size of each batch. Defaults to None.
        mirrored (bool, optional): whether to use dihedral predictions (every simmetry). Defaults to False.

    Returns:
        np.ndarray: numpy array with dimensions (w, h, num_classes), containing smooth predictions
    """
    width, height, _ = image.shape
    padded = pad_image(image=image, tile_size=tile_size, subdivisions=subdivisions)
    padded_width, padded_height, _ = padded.shape
    padded_variants = rotate_and_mirror(padded) if mirrored else [padded]
    spline = _spline_2d(window_size=tile_size, power=2)

    results = []
    for img in padded_variants:
        canvas = np.zeros((padded_width, padded_height, num_classes))
        for coords, batch in windowed_generator(padded_image=img,
                                                window_size=tile_size,
                                                subdivisions=subdivisions,
                                                batch_size=batch_size):
            pred_batch = prediction_fn(batch)
            pred_batch = [tile * spline for tile in pred_batch]
            canvas = reconstruct(canvas, tile_size=tile_size, coords=coords, predictions=pred_batch)
        canvas /= (subdivisions**2)
        results.append(canvas)

    padded_result = undo_rotate_and_mirror(results) if mirrored else results[0]
    prediction = unpad_image(padded_result, tile_size=tile_size, subdivisions=subdivisions)
    return prediction[:width, :height, :]
