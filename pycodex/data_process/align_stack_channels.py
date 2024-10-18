import tifffile
import numpy as np 
import cv2 
import os 
import logging
import re
from tqdm import tqdm
from scipy.ndimage import map_coordinates
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


PATH_TO_QPTIFF = 'cHL_TMA_raw_qptiff'

CORE_POSITION = {'Cycle1': [20258, 29161, 16114,25000],
'Cycle2': [19286,29168, 15187,24987],
'Cycle3': [18929,29178, 14805,24981],
'Cycle4': [18660,29161, 14577,24993],
'Cycle5': [18702,29165, 14629,24990]}


CHANNEL_NAME = ['DAPI', 'CD20', 'Pax5', 'CD3', 'CD8', 'CD4', 'FoxP3', 'CD68', 'CD163', 'CD11c', 'CD31']



OUTPUT_HEIGHT = 4161
OUTPUT_WIDTH = 4144


def list_files(directory: str):
    paths = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            paths.append(os.path.join(root, file))
    return paths


def read_dapi(path: str):
    cycle = re.search(r'(Cycle\d+)', path).group(1)
    x2 = CORE_POSITION[cycle][0]
    y2 = CORE_POSITION[cycle][1]
    x1 = CORE_POSITION[cycle][2]
    y1 = CORE_POSITION[cycle][3]
    dapi = tifffile.TiffFile(path).series[0].pages[0].asarray()[y1:y2,x1:x2]
    return dapi


def register_dapi(src, dst, sift, matcher, outname):
    src = (src/256).astype('uint8')[::8, ::8]
    dst = (dst/256).astype('uint8')[::8, ::8]
    kp1, des1 = sift.detectAndCompute(src, None)
    kp2, des2 = sift.detectAndCompute(dst, None)
    matches = matcher.knnMatch(des1, des2, k = 2)
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)
    # Extract matched keypoints
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        # draw match image
    match_img = cv2.drawMatches(img1 = src,
                            keypoints1=kp1,
                            img2 = dst,
                            keypoints2=kp2,
                            matches1to2=good_matches,
                            outImg=None,
                            flags = cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    tifffile.imwrite(f'registration_output/{outname}.png', match_img, compression='adobe_deflate')

    # Compute affine transformation
    A = cv2.estimateAffinePartial2D(src_pts, dst_pts)[0]
    A[0,2] *= 8
    A[1,2] *= 8
    A_inverse = np.linalg.inv(np.vstack((A, [0,0,1])))
    return A, A_inverse

        
def apply_affine_transformation(source_image, A_inverse, output_shape):
    
    dest_y, dest_x = np.indices(output_shape)

    dest_coords = np.stack([dest_x.ravel(), dest_y.ravel(), np.ones(dest_x.size)])

    source_coords = A_inverse @ dest_coords

    source_x = source_coords[0, :].reshape(output_shape)

    source_y = source_coords[1, :].reshape(output_shape)

    transformed_image = map_coordinates(source_image, [source_y, source_x], order = 1, mode = 'constant', cval=0)

    transformed_image = np.clip(transformed_image, 0, 65535).astype(np.uint16)

    return transformed_image


def main():
    qptiff_paths = [i for i in list_files(PATH_TO_QPTIFF) if '.raw.qptiff' in i]
    
    DAPI_list = []

    for path in qptiff_paths:
        logging.info(f'Reading DAPI from {path}.')
        dapi = read_dapi(path)
        DAPI_list.append(dapi)
    logging.info(f'Finished Reading all DAPI.')

    image_stack = []

    image_stack.append(DAPI_list[0])

    image_stack.append(tifffile.TiffFile(qptiff_paths[0]).series[0].pages[1].asarray()[25000:29161, 16114:20258])

    image_stack.append(tifffile.TiffFile(qptiff_paths[0]).series[0].pages[3].asarray()[25000:29161, 16114:20258])

    #logging.info(f'{image_stack[0].shape} \n {image_stack[1].shape} \n {image_stack[2].shape} \n')

    for i in tqdm(range(4)):
        sift = cv2.SIFT_create(nfeatures=1000)
        matcher = cv2.BFMatcher()
        logging.info(f'Calculating affine transformation matrix from cycle{i+2} to cycle1...')
        A, A_inverse = register_dapi(DAPI_list[i + 1], DAPI_list[0], sift, matcher, f'cycle{i+2}_to_cycle1')
        logging.info(f'Finished calculation')
        
        cycle_file = qptiff_paths[i+1]
        x2 = CORE_POSITION[f'Cycle{i+2}'][0]
        y2 = CORE_POSITION[f'Cycle{i+2}'][1]
        x1 = CORE_POSITION[f'Cycle{i+2}'][2]
        y1 = CORE_POSITION[f'Cycle{i+2}'][3]
        cy3 = tifffile.TiffFile(cycle_file).series[0].pages[1].asarray()[y1:y2,x1:x2]
        cy5 = tifffile.TiffFile(cycle_file).series[0].pages[3].asarray()[y1:y2,x1:x2]
        logging.info(f'Finished loading cy3 and cy5 of the cycle')


        
        # Apply affine transformation using cv2.warpAffine
        cy3_out = apply_affine_transformation(cy3, A_inverse, (OUTPUT_HEIGHT, OUTPUT_WIDTH))
        cy5_out = apply_affine_transformation(cy5, A_inverse, (OUTPUT_HEIGHT, OUTPUT_WIDTH))
        logging.info(f'Finished transforming cy3 and cy5.')
        tifffile.imwrite(f'registration_output/{i+2}_cy3.tiff', cy3_out)
        tifffile.imwrite(f'registration_output/{i+2}_cy5.tiff', cy5_out)
        #logging.info(f'cy3 shape: {cy3_out.shape}; cy5 shape: {cy5_out.shape}')
        
        image_stack.append(cy3_out)
        image_stack.append(cy5_out)

    aligned_check = np.stack(image_stack, axis=0)

    logging.info(f'Finished stacking the channels.')

    logging.info(f'Writing OMETIFF...')

    with tifffile.TiffWriter('aligned_channels.ome.tiff', bigtiff=True) as tif:
        options = dict(tile=(512, 512), compression='adobe_deflate', metadata = {'axes': 'CYX', 'Channel': {'Name': CHANNEL_NAME}})
        tif.write(aligned_check, subifds = 4, **options)
        tif.write(aligned_check[:, ::2, ::2], subfiletype = 1, **options)
        tif.write(aligned_check[:, ::4, ::4], subfiletype = 1, **options)
        tif.write(aligned_check[:, ::8, ::8], subfiletype = 1, **options)
        tif.write(aligned_check[:, ::16, ::16], subfiletype = 1, **options)
    logging.info('Finished.')



                






if __name__ == '__main__':
    main()

    





















