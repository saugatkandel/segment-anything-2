from pathlib import Path

import cucim.skimage as cskimage
import cupy
import joblib
import numpy as np
from copick.models import CopickRun
from tqdm.notebook import tqdm


def _check_input(non_copick_input_dir, use_copick_masks, copick_session_id):
    if non_copick_input_dir is None and not use_copick_masks:
        raise ValueError("Either non_copick_input_dir or use_copick_masks must be provided")
    if use_copick_masks and copick_session_id is None:
        raise ValueError("Session ID must be provided if use_copick_masks is True")
    return True


def get_data_from_copick(run: CopickRun, session_id: int, z_trim: int = 10):
    """
    Retrieve segmentation data from a Copick run.

    Parameters
    ----------
    run : CopickRun
        The Copick run object.
    session_id : int
        The session ID for the Copick run.
    z_trim : int, optional
        Number of slices to trim from the z-axis. Default is 10.

    Returns
    -------
    cupy.ndarray or None
        The volume mask as a cupy array, or None if no segmentations are found.
    """
    organelle_segs = run.get_segmentations("cryoSAM2", session_id=str(session_id))
    if len(organelle_segs) == 0:
        return None
    elif len(organelle_segs) > 1:
        raise ValueError("More than one segmentation found")

    vol_mask = cupy.asarray(organelle_segs[0].numpy())
    return vol_mask


def get_data_from_local_pkl(run: CopickRun, non_copick_input_dir: str | Path):
    """
    Retrieve data from local pickle files.

    Parameters
    ----------
    run : CopickRun
        The Copick run object.
    non_copick_input_dir : str or Path
        Directory path for non-copick input files.

    Returns
    -------
    tuple
        A tuple containing auto_mask and vol_mask.
    """
    auto_mask_fname = Path(non_copick_input_dir) / f"auto_masks_{run.name}.pkl"
    vol_mask_fname = Path(non_copick_input_dir) / f"vol_mask_{run.name}.pkl"
    if not vol_mask_fname.exists():
        return None, None
    auto_mask = joblib.load(auto_mask_fname)
    vol_mask = joblib.load(vol_mask_fname)
    vol_mask = cupy.asarray(vol_mask)
    return auto_mask, vol_mask


def calculate_statistics_for_daniel_picks(
    runs_to_analyze: list[CopickRun],
    picks_coords_per_run: dict,
    non_copick_input_dir: str | Path | None = None,
    use_copick_masks: bool = False,
    copick_session_id: int | None = None,
    z_trim: int = 10,
):
    """
    Calculate statistics for given runs and coordinates.

    Parameters
    ----------
    runs_to_analyze : list
        List of runs to analyze.
    picks_coords_per_run : dict
        Dictionary mapping run names to their respective coordinates.
        The typical case is to use a dictionary containing Daniel's picks for lysosomes.
    non_copick_input_dir : str, optional
        Directory path for non-copick input files. Default is None. Either this or use_copick_masks must be provided.
    use_copick_masks : bool, optional
        Flag to indicate whether to use copick masks. Default is False. Session ID must be provided if True.
    copick_session_id : int, optional
        Session ID for copick masks. The session id must be provided if use_copick_masks is True.
    z_trim : int, optional
        Number of slices to trim from the z-axis. Default is 10.

    Returns
    -------
    results : dict
        Dictionary containing calculated statistics for each run.
    auto_masks : dict
        Dictionary containing auto masks for each run.
    """
    results = {}
    auto_masks = {}
    per_tomo_true_positives = {}
    per_tomo_false_positives = {}
    per_tomo_false_negatives = {}

    _check_input(non_copick_input_dir, use_copick_masks, copick_session_id)

    for run in tqdm(runs_to_analyze):
        results[run.name] = []
        per_tomo_true_positives[run.name] = 0
        per_tomo_false_positives[run.name] = 0
        per_tomo_false_negatives[run.name] = 0

        if run.name in picks_coords_per_run:
            coords = picks_coords_per_run[run.name]
        else:
            coords = np.empty((0, 3))

        if use_copick_masks:
            vol_mask = get_data_from_copick(run, copick_session_id, z_trim)
        else:
            auto_mask, vol_mask = get_data_from_local_pkl(run, non_copick_input_dir)
            auto_masks[run.name] = auto_mask

        # if coords is None:
        #    if vol_mask is not None:
        #        if vol_mask.max() > 0:
        #            unique_labels = np.unique(vol_mask)
        #            nonzero_unique_labels = unique_labels[unique_labels != 0]
        #
        #           per_tomo_false_positives[run.name] = len(nonzero_unique_labels)
        #   continue
        if vol_mask is None:
            per_tomo_false_negatives[run.name] = len(coords)
            continue

        vol_mask_trimmed = vol_mask[z_trim:-z_trim]
        unique_labels = np.unique(vol_mask_trimmed)
        nonzero_unique_labels = unique_labels[unique_labels != 0]

        if len(nonzero_unique_labels) == 0:
            per_tomo_false_negatives[run.name] = len(coords)
            continue

        _preds_for_ground_truths = vol_mask[*np.floor(coords.T).astype("int")]

        labels_seen = []
        for cidx, label in enumerate(_preds_for_ground_truths):
            if label == 0 or label in labels_seen:
                per_tomo_false_negatives[run.name] += 1
                continue
            labels_seen.append(label)
            per_tomo_true_positives[run.name] += 1
            coord = coords[cidx]

            vol_with_label = (vol_mask_trimmed == label).astype("int")
            centroid = cskimage.measure.regionprops(vol_with_label)[0].centroid

            centroid = np.array(centroid) + np.array([z_trim, 0, 0])

            z, y, x = np.floor(coord).astype("int")

            _zmin = max(0, z - 5)
            _zmax = min(vol_with_label.shape[0], z + 5)
            vol_trunc = vol_with_label[_zmin:_zmax].astype("int")

            # vol_trunc = vol_mask[z - 5 : z + 5]
            # vol_label_only = (vol_trunc == label).astype("int")
            # region = vol_label_only.max(axis=0)
            region = vol_trunc.max(axis=0)

            rprops = cskimage.measure.regionprops(region)

            if len(rprops) > 1:
                raise ValueError("More than one region found")

            axmin = rprops[0].axis_minor_length
            axmaj = rprops[0].axis_major_length
            orientation = rprops[0].orientation
            _result_dict = {
                "coord": coord,
                "label": int(label),
                "sam2_centroid": centroid,
                "axmin": axmin,
                "axmaj": axmaj,
                "orientation": orientation,
            }
            results[run.name].append(_result_dict)

        unique_labels = np.unique(vol_mask)
        nonzero_unique_labels = unique_labels[unique_labels != 0]
        per_tomo_false_positives[run.name] = len(nonzero_unique_labels) - len(labels_seen)
    evaluations = (per_tomo_true_positives, per_tomo_false_positives, per_tomo_false_negatives)
    return results, auto_masks, evaluations


def calculate_statistics_for_all_picks(
    runs_to_analyze: list[CopickRun],
    non_copick_input_dir: str | Path | None = None,
    use_copick_masks: bool = False,
    copick_session_id: int | None = None,
    z_trim: int = 10,
):
    """
    Calculate statistics for given runs and coordinates.

    Parameters
    ----------
    runs_to_analyze : list
        List of runs to analyze.
    non_copick_input_dir : str, optional
        Directory path for non-copick input files. Default is None. Either this or use_copick_masks must be provided.
    use_copick_masks : bool, optional
        Flag to indicate whether to use copick masks. Default is False. Session ID must be provided if True.
    copick_session_id : int, optional
        Session ID for copick masks. The session id must be provided if use_copick_masks is True.
    z_trim : int, optional
        Number of slices to trim from the z-axis. Default is 10.

    Returns
    -------
    results : dict
        Dictionary containing calculated statistics for each run.
    auto_masks : dict
        Dictionary containing auto masks for each run.
    """
    results = {}
    auto_masks = {}

    _check_input(non_copick_input_dir, use_copick_masks, copick_session_id)

    for run in tqdm(runs_to_analyze):
        print("Analyzing run:", run.name)
        results[run.name] = []
        if use_copick_masks:
            vol_mask = get_data_from_copick(run, copick_session_id, z_trim)
            if vol_mask is None:
                continue
        else:
            auto_mask, vol_mask = get_data_from_local_pkl(run, non_copick_input_dir)
            auto_masks[run.name] = auto_mask
            if vol_mask is None:
                continue

        vol_mask_trimmed = vol_mask[z_trim:-z_trim]
        unique_labels = np.unique(vol_mask_trimmed)
        nonzero_unique_labels = unique_labels[unique_labels != 0]

        if len(nonzero_unique_labels) == 0:
            continue

        for label in nonzero_unique_labels:
            vol_with_label = (vol_mask_trimmed == label).astype("int")

            centroid = cskimage.measure.regionprops(vol_with_label)[0].centroid
            z, y, x = np.floor(centroid).astype("int")

            _zmin = max(0, z - 5)
            _zmax = min(vol_with_label.shape[0], z + 5)
            vol_trunc = vol_with_label[_zmin:_zmax].astype("int")
            region = vol_trunc.max(axis=0)

            rprops = cskimage.measure.regionprops(region)

            if len(rprops) == 0:
                continue
            if len(rprops) > 1:
                raise ValueError("More than one region found")

            axmin = rprops[0].axis_minor_length
            axmaj = rprops[0].axis_major_length
            orientation = rprops[0].orientation
            _result_dict = {
                "centroid": centroid + np.array([z_trim, 0, 0]),
                "label": int(label),
                "axmin": axmin,
                "axmaj": axmaj,
                "orientation": orientation,
            }

            results.setdefault(run.name, [])
            results[run.name].append(_result_dict)
    return results, auto_masks
