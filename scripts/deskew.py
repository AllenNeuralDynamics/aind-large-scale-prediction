"""
Sample script for applying dispim deskewing
"""

import multiprocessing
from time import time
from typing import Tuple

import dask
import dask.array as da
import matplotlib.pyplot as plt
import numpy as np
import tifffile as tif
from dask.distributed import Client, LocalCluster, performance_report
from dask_image.ndinterp import affine_transform as dask_affine_transform
from scipy.ndimage import affine_transform


def ceil_to_mulitple(x, base: int = 4):
    """rounds up to the nearest integer multiple of base

    Parameters
    ----------
    x : scalar or np.array
        value/s to round up from
    base : int, optional
        round up to multiples of base (the default is 4)
    Returns
    -------
    scalar or np.array:
        rounded up value/s

    """

    return (
        np.int32(base) * np.ceil(np.array(x).astype(np.float32) / base)
    ).astype(np.int32)


def get_transformed_corners(
    aff: np.ndarray, vol_or_shape, zeroindex: bool = True
):
    """Input
    aff: an affine transformation matrix
    vol_or_shape: a numpy volume or shape of a volume.

    This function will return the positions of the corner points of the volume (or volume with
    provided shape) after applying the affine transform.
    """
    # get the dimensions of the array.
    # see whether we got a volume
    if np.array(vol_or_shape).ndim == 3:
        d0, d1, d2 = np.array(vol_or_shape).shape
    elif np.array(vol_or_shape).ndim == 1:
        d0, d1, d2 = vol_or_shape
    else:
        raise ValueError
    # By default we calculate where the corner points in
    # zero-indexed (numpy) arrays will be transformed to.
    # set zeroindex to False if you want to perform the calculation
    # for Matlab-style arrays.
    if zeroindex:
        d0 -= 1
        d1 -= 1
        d2 -= 1
    # all corners of the input volume (maybe there is
    # a more concise way to express this with itertools?)
    corners_in = [
        (0, 0, 0, 1),
        (d0, 0, 0, 1),
        (0, d1, 0, 1),
        (0, 0, d2, 1),
        (d0, d1, 0, 1),
        (d0, 0, d2, 1),
        (0, d1, d2, 1),
        (d0, d1, d2, 1),
    ]
    corners_out = list(map(lambda c: aff @ np.array(c), corners_in))
    corner_array = np.concatenate(corners_out).reshape((-1, 4))
    # print(corner_array)
    return corner_array


def get_output_dimensions(aff: np.ndarray, vol_or_shape):
    """given an 4x4 affine transformation matrix aff and
    a 3d input volume (numpy array) or volumen shape (iterable with 3 elements)
    this function returns the output dimensions required for the array after the
    transform. Rounds up to create an integer result.
    """
    corners = get_transformed_corners(aff, vol_or_shape, zeroindex=True)
    # +1 to avoid fencepost error
    dims = np.max(corners, axis=0) - np.min(corners, axis=0) + 1
    dims = ceil_to_mulitple(dims, 2)
    return dims[:3].astype(np.int32)


# def plot_all(imlist, backend: str = "matplotlib"):
#     """ given an iterable of 2d numpy arrays (images),
#         plots all of them in order.
#         Will add different backends (Bokeh) later """
#     if backend == "matplotlib":
#         for im in imlist:
#             plt.imshow(im)
#             plt.show()
#     else:
#         pass


def get_projection_montage(
    vol: np.ndarray, gap: int = 10, proj_function=np.max
) -> np.ndarray:
    """given a volume vol, creates a montage with all three projections (orthogonal views)

    Parameters
    ----------
    vol : np.ndarray
        input volume
    gap : int, optional
        gap between projections in montage (the default is 10 pixels)
    proj_function : Callable, optional
        function to create the projection (the default is np.max, which performs maximum projection)

    Returns
    -------
    np.ndarray
        the montage of all projections
    """

    assert len(vol.shape) == 3, "only implemented for 3D-volumes"
    nz, ny, nx = vol.shape
    m = np.zeros((ny + nz + gap, nx + nz + gap), dtype=vol.dtype)
    m[:ny, :nx] = proj_function(vol, axis=0)
    m[ny + gap :, :nx] = np.max(vol, axis=1)
    m[:ny, nx + gap :] = np.max(vol, axis=2).transpose()
    return m


def get_dispim_config():
    """
    Returns dispim configuration
    of the microscope
    """
    return {
        "resolution": {"x": 0.298, "y": 0.298, "z": 0.176},
        "angle_degrees": 45,  # with respect to xy
    }


def shear_angle_to_shear_factor(angle_in_degrees):
    """
    Converts a shearing angle into a shearing factor
    Parameters
    ----------
    angle_in_degrees: float
    Returns
    -------
    float
    """
    return 1.0 / np.tan((90 - angle_in_degrees) * np.pi / 180)


def shear_factor_to_shear_angle(shear_factor):
    """
    Converts a shearing angle into a shearing factor
    Parameters
    ----------
    shear_factor: float
    Returns
    -------
    float
    """
    return -np.atan(1.0 / shear_factor) * 180 / np.pi + 90


def create_translation_in_centre(
    image_shape: tuple, orientation: int = -1
) -> np.matrix:
    """
    Creates a translation from the center

    Parameters
    ----------
    image_shape: tuple
        Image shape

    orientation: int
        Orientation. -1 to move from the center,
        1 to come back to original center

    Returns
    -------
    np.matrix
        Matrix with the image transformation
    """
    centre = np.array(image_shape) / 2
    shift_transformation = np.eye(4, dtype=np.float32)
    shift_transformation[:3, 3] = orientation * centre

    return shift_transformation


def rot_around_y(angle_deg: float) -> np.ndarray:
    """create affine matrix for rotation around y axis

    Parameters
    ----------
    angle_deg : float
        rotation angle in degrees

    Returns
    -------
    np.ndarray
        4x4 affine rotation matrix
    """
    arad = angle_deg * np.pi / 180.0
    roty = np.array(
        [
            [np.cos(arad), 0, np.sin(arad), 0],
            [0, 1, 0, 0],
            [-np.sin(arad), 0, np.cos(arad), 0],
            [0, 0, 0, 1],
        ]
    )
    return roty


def create_rotation_transformation(angle_radians: float) -> np.matrix:
    """
    Rotation in Y

    Parameters
    ----------
    angle_radians: float
        Angle in radians for the rotation

    Returns
    -------
    np.matrix
        Matrix with the rotation transformation
        around y
    """
    rotation_transformation = np.eye(4, dtype=np.float32)
    rotation_transformation[0][0] = np.cos(angle_radians)
    rotation_transformation[0][2] = np.sin(angle_radians)
    rotation_transformation[2][0] = -rotation_transformation[0][2]
    rotation_transformation[2][2] = rotation_transformation[0][0]

    return rotation_transformation


def create_dispim_transform(
    image_data_shape: Tuple[int, ...], config: dict, scale: bool
) -> Tuple[np.matrix, Tuple[int, ...]]:
    """
    Creates the dispim transformation following
    the provided parameters in the configuration

    Parameters
    ----------
    image_data_shape: Tuple[int, ...]
        Image data shape

    config: dict
        Configuration dictionary

    scale: bool
        If true, we make the data isotropy scaling
        Z to XY

    Returns
    -------
    Tuple[np.matrix, Tuple[int, ...]]
        Matrix with the affine transformation
        and the output shape for the transformed
        image
    """
    shear_factor = shear_angle_to_shear_factor(
        config["shift"] * config["angle_degrees"]
    )

    # num_imaged_images = image_data.shape[0] * config["zy_pixel_size"]

    shear_matrix = np.array(
        [[1, 0, shear_factor, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    )
    print("Shear matrix: ", shear_matrix)

    # new_dz = np.sin(config["angle_degrees"] * np.pi / 180.0) * config["resolution"]["z"]
    # scale_factor_z = (new_dz / config["resolution"]["x"]) * 1.0

    scale_matrix = np.eye(4, dtype=np.float32)

    if scale:
        scale_matrix[0, 0] = (
            config["resolution"]["y"] / config["resolution"]["z"]
        )  # scale_factor_z
        # scale_matrix[1, 1] = config["resolution"]["y"] / config["resolution"]["z"] #scale_factor_z
        # scale_matrix[2, 2] = config["resolution"]["x"] / config["resolution"]["z"] #scale_factor_z

    print("Scale matrix: ", scale_matrix)

    # Rotation to coverslip
    translation_transformation_center = create_translation_in_centre(
        image_shape=image_data_shape, orientation=-1
    )
    print("Translation from origin: ", translation_transformation_center)

    # rotation_transformation = create_rotation_transformation(config=config["angle_radians"])
    # print("Rotation matrix: ", rotation_transformation)

    # Axis order X:0, Y:1, Z:2 - Using yaw-pitch-roll transform ZYX
    # rotation_matrix = R_yaw @ R_pitch @ R_roll -- Here we're only
    # applying pitch and identities for yaw and roll
    # new_axis_order = np.argsort(rotation_transformation[0, :3])

    shift_shear_rot = (
        # rotation_transformation
        scale_matrix
        @ shear_matrix
        @ translation_transformation_center
    )

    output_shape_after_rot = get_output_dimensions(
        shift_shear_rot, image_data_shape
    )

    back_from_translation = create_translation_in_centre(
        output_shape_after_rot, orientation=1
    )

    final_transform = back_from_translation @ shift_shear_rot

    return final_transform, output_shape_after_rot


def create_dispim_config(multiscale: int, camera: int) -> dict:
    """
    Creates the dispim configuration dictionary
    for deskewing the data

    Parameters
    ----------
    multiscale: int
        Multiscale we want to use to deskew

    camera: int
        Camera the data was acquired with. The dispim
        has two cameras in an angle of 45 and -45 degrees

    Returns
    -------
    dict
        Dictionary with the information for deskewing the data
    """
    config = get_dispim_config()
    config["resolution"]["x"] = config["resolution"]["x"] * (2**multiscale)
    config["resolution"]["y"] = config["resolution"]["y"] * (2**multiscale)
    config["resolution"]["z"] = config["resolution"]["z"] * (2**multiscale)

    shift = 1

    if camera:
        shift = -1
        print("Changing shift: ", shift)

    # Shifting
    config["shift"] = shift

    # Angle in radians
    config["angle_radians"] = np.deg2rad(config["angle_degrees"])

    # Z stage movement in um
    config["zstage"] = (config["resolution"]["z"]) / np.sin(
        config["angle_radians"]
    )  # 0.299401

    # XY pixel size ratio in um
    config["xy_pixel_size"] = (
        config["resolution"]["x"] / config["resolution"]["y"]
    )  # 0.1040

    # ZY pixel size in um
    config["yz_pixel_size"] = (
        config["resolution"]["y"] / config["resolution"]["z"]
    )

    return config


def apply_affine_dask(
    image_data: da.array,
    affine_transformation: np.matrix,
    output_shape: Tuple[int, ...],
) -> da.array:
    """
    Applies a lazy affine transformation

    Parameters
    ----------
    image_data: da.array
        Lazy image data

    affine_transformation: np.matrix
        Affine transformation that will
        be applied to the data

    output_shape: Tuple[int, ...]
        Output shape that the data will
        have after applying the transformation

    Returns
    -------
    np.array
        Array with the data in memory
    """
    transformed_image = dask_affine_transform(
        image=image_data,
        matrix=np.linalg.inv(affine_transformation.copy()),
        output_shape=output_shape,
        output_chunks=image_data.chunksize,
        order=1,
    )

    dask.config.set(
        {
            # "tcp-timeout": "300s",
            # "array.chunk-size": "384MiB",
            # "distributed.comm.timeouts": {"connect": "300s", "tcp": "300s",},
            "distributed.scheduler.bandwidth": 100000000,
            # "managed_in_memory",#
            "distributed.worker.memory.rebalance.measure": "managed",
            "distributed.worker.memory.target": False,  # 0.85,
            "distributed.worker.memory.spill": False,  # False,#
            "distributed.worker.memory.pause": False,  # False,#
            "distributed.worker.memory.terminate": False,  # False, #
            # 'distributed.scheduler.unknown-task-duration': '15m',
            # 'distributed.scheduler.default-task-durations': '2h',
        }
    )

    print(f"# of CPUs {multiprocessing.cpu_count()}")
    n_workers = 4
    threads_per_worker = 1

    cluster = LocalCluster(
        n_workers=n_workers,
        threads_per_worker=threads_per_worker,
        processes=True,
        memory_limit="0.5GB",
    )
    client = Client(cluster)
    print(client)

    dask_report_file = "report_file.html"
    print("Computing the affine")

    start_time = time()

    with performance_report(filename=dask_report_file):
        transformed_image = dask.optimize(transformed_image)[0]
        transformed_image_computed = transformed_image.compute()

    client.close()

    end_time = time()
    print(f"Time needed to convert data: {end_time-start_time}")
    print(f"Output shape: {transformed_image_computed.shape}")

    return transformed_image_computed


def transform_in_memory(
    dataset_path: str, multiscale: int, camera: int, make_isotropy_voxels: bool
):
    """
    Applies an affine transformation
    to a zarr image dataset.

    Parameters
    ----------
    dataset_path: str
        Path where the data is stored in the
        cloud

    multiscale: int
        Multiscale we will load

    camera: int
        Camera the dataset was acquired with

    make_isotropy_voxels: bool
        Makes the voxel size isotropic
    """
    original_image_dask = da.from_zarr(f"{dataset_path}/{multiscale}")
    sample_block = original_image_dask.compute()[0, 0, ...]
    print(f"Sample dispim block {sample_block.shape}")

    start = time()
    # Creates diSPIM config
    config = create_dispim_config(multiscale=multiscale, camera=camera)

    # Creates the image transformation
    final_transform, output_shape = create_dispim_transform(
        image_data_shape=sample_block.shape,
        config=config,
        scale=make_isotropy_voxels,
    )

    # Computes the affine transform
    transformed_volume = affine_transform(
        input=sample_block,
        matrix=np.linalg.inv(final_transform),
        output_shape=output_shape,
        mode="constant",
        order=1,  # Bilinear interpolation -> balance between speed and quality
    )

    end = time()
    print(f"Total time: {end - start}")
    tif.imwrite("deskewed_in_memory.tif", transformed_volume)


def transform_with_dask(
    dataset_path: str, multiscale: int, camera: int, make_isotropy_voxels: bool
):
    """
    Applies an affine transformation
    to a zarr image dataset.

    Parameters
    ----------
    dataset_path: str
        Path where the data is stored in the
        cloud

    multiscale: int
        Multiscale we will load

    camera: int
        Camera the dataset was acquired with

    make_isotropy_voxels: bool
        Makes the voxel size isotropic
    """
    original_image_dask = da.from_zarr(f"{dataset_path}/{multiscale}")[
        0, 0, 5000:5500, 700:1100, 800:1200
    ]

    start = time()
    # Creates diSPIM config
    config = create_dispim_config(multiscale=multiscale, camera=camera)

    # Creates the image transformation
    final_transform, output_shape = create_dispim_transform(
        image_data_shape=original_image_dask.shape,
        config=config,
        scale=make_isotropy_voxels,
    )

    # Computes the affine transform
    transformed_volume = apply_affine_dask(
        image_data=original_image_dask,
        affine_transformation=final_transform,
        output_shape=output_shape,
    )

    end = time()
    print(f"Total time: {end - start}")
    tif.imwrite("deskewed_dask.tif", transformed_volume)


def main():
    """
    Main function
    """

    dataset_path = "s3://aind-open-data/diSPIM_685890_2023-06-29_14-39-56/diSPIM.zarr/647_D1_X_0001_Y_0001_Z_0000_ch_488.zarr"
    # dataset_path = "s3://aind-open-data/diSPIM_662960_2023-08-10_13-20-37/diSPIM.zarr/960_ventr_X_0003_Y_0000_Z_0000_ch_561.zarr"
    # dataset_path = "s3://aind-open-data/HCR_663983_2023-10-10_09-35-12/SPIM.ome.zarr/983_diSPIM_X_0002_Y_0000_Z_0000_ch_488.zarr"
    multiscale = 0
    camera = 1
    make_isotropy_voxels = False
    # transform_in_memory(dataset_path, multiscale, camera, make_isotropy_voxels)
    transform_with_dask(dataset_path, multiscale, camera, make_isotropy_voxels)


def plot_array_images(imgs, title, top=1.2):
    """
    Plots the array image
    """
    fig = plt.figure(figsize=(10, 10))
    fig.suptitle(title, fontsize=10)
    for k, img in enumerate(imgs):
        vmin, vmax = np.percentile(img, (0.1, 99))
        plt.subplot(1, 3, k + 1)
        plt.imshow(img, vmin=vmin, vmax=vmax, cmap="gray")

    fig.tight_layout()
    # fig.subplots_adjust(top=top)
    plt.show()


def visualize_data():
    """
    Small function to visualize data
    """

    gaussian_psf_path = "/Users/camilo.laiton/repositories/dispim_psf_estimation/initial_psf.tif"
    gaussian_psf = tif.imread(gaussian_psf_path)

    dataset_path = "s3://aind-open-data/HCR_Christian-ME-New-Lasers_2023-10-05_11-37-41_fused/channel_488.zarr"
    multiscale = 0

    lazy_image = da.from_zarr(f"{dataset_path}/{multiscale}")[0, 0, ...]
    z, y, x = lazy_image.shape
    print("Image shape: ", lazy_image.shape)
    s = 128
    image_data = lazy_image[
        (z // 2) - s : (z // 2),
        (y // 2) - s : (y // 2),
        (x // 2) - s * 2 : (x // 2) - s,
    ].compute()

    print("block shape: ", image_data.shape)

    for axis in range(0, 3):
        axis_name = ["XY", "ZX", "ZY"]
        plot_array_images(
            imgs=[
                np.max(image_data, axis=axis),
                np.max(gaussian_psf, axis=axis),
            ],
            title=f"Gaussian PSF vs Real PSF - {axis_name[axis]}",
        )


if __name__ == "__main__":
    visualize_data()
