import matplotlib.pyplot as plt
import zarr, cv2, torch
import numpy as np

def get_tomogram(run, voxel_size = 10, algorithm = 'denoised'):

    print(f'Getting {algorithm} Tomogram with {voxel_size} A voxel size for the associated runID: {run.name}')

    tomogram = run.get_voxel_spacing(voxel_size).get_tomogram(algorithm)

    # Access the data
    group = zarr.open(tomogram.zarr())
    arrays = list(group.arrays())

    # Return Volume
    return arrays[0][1][:]

def get_coordinates(run, name = 'lysosome', voxel_size = 10):

    points = run.get_picks(object_name = name)[0].points

    # Initialize an array to store the coordinates
    nPoints = len(points)                      # Number of points retrieved
    coordinates = np.zeros([len(points), 3])   # Create an empty array to hold the (z, y, x) coordinates

    # Iterate over all points and convert their locations to coordinates in voxel space
    for ii in range(nPoints):
        coordinates[ii,] = [points[ii].location.z / voxel_size,   # Scale z-coordinate by voxel size
                            points[ii].location.y / voxel_size,   # Scale y-coordinate by voxel size
                            points[ii].location.x / voxel_size]   # Scale x-coordinate by voxel size

    return coordinates

def project_tomogram(vol, zSlice = None, deltaZ = None):
    """
    Projects a tomogram along the z-axis.
    
    Parameters:
    vol (np.ndarray): 3D tomogram array (z, y, x).
    zSlice (int, optional): Specific z-slice to project. If None, project along all z slices.
    deltaZ (int, optional): Thickness of slices to project. Used only if zSlice is specified. If None, project a single slice.

    Returns:
    np.ndarray: 2D projected tomogram.
    """    

    if zSlice is not None:
        # If deltaZ is specified, project over zSlice to zSlice + deltaZ
        if deltaZ is not None:
            zStart = max(zSlice, 0)
            zEnd = min(zSlice + deltaZ, vol.shape[0])  # Ensure we don't exceed the volume size
            projection = np.sum(vol[zStart:zEnd,], axis=0)  # Sum over the specified slices
        else:
            # If deltaZ is not specified, project just a single z slice
            projection = vol[zSlice,]
    else:
        # If zSlice is None, project over the entire z-axis
        projection = np.sum(vol, axis=0)

    # test_data_norm = (test_data - test_data.min()) / (test_data.max() - test_data.min())
    # image = np.repeat(test_data_norm[..., None], 3, axis=2)           
    
    return projection

##################### Meta FAIR Utility Functions #####################

# def show_mask(mask, ax, random_color=False, borders = True):
#     if random_color:
#         color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
#     else:
#         color = np.array([30/255, 144/255, 255/255, 0.6])
#     h, w = mask.shape[-2:]
#     mask = mask.astype(np.uint8)
#     mask_image =  mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
#     if borders:
#         contours, _ = cv2.findContours(mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
#         # Try to smooth contours
#         contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
#         mask_image = cv2.drawContours(mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2) 
#     ax.imshow(mask_image)

def show_mask(mask, ax, obj_id=None, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))    

def show_masks(image, masks, scores, point_coords=None, box_coords=None, input_labels=None, borders=True):
    for i, (mask, score) in enumerate(zip(masks, scores)):
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        show_mask(mask, plt.gca(), borders=borders)
        if point_coords is not None:
            assert input_labels is not None
            show_points(point_coords, input_labels, plt.gca())
        if box_coords is not None:
            # boxes
            show_box(box_coords, plt.gca())
        if len(scores) > 1:
            plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
        plt.axis('off')
        plt.show()

def show_anns(anns, borders=True):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.5]])
        img[m] = color_mask 
        if borders:
            import cv2
            contours, _ = cv2.findContours(m.astype(np.uint8),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
            # Try to smooth contours
            contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
            cv2.drawContours(img, contours, -1, (0,0,1,0.4), thickness=1) 

    ax.imshow(img)        

def show_tomo_frame(tomo, frame_id, ax):
    frame = torch.tensor(tomo[frame_id], device="cpu")
    if frame.ndim == 3:
        frame = frame[0]

    ax.imshow(frame, cmap="gray")
    ax.set_title(f"Frame {frame_id}")    