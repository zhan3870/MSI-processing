import numpy as np
import tensorflow as tf
import os
from PIL import Image
import time
from sys import argv
import tifffile as tiff
import glob
import re


def denoise2D(im, imm):
    im = im/imm-0.5
    with tf.compat.v1.Session() as sess:
        in_t = im[np.newaxis,...,np.newaxis]
        input = graph.get_operations()[0]
        output = graph.get_operations()[161]
        output_t = sess.run("activation/Tanh:0", feed_dict={"img:0": in_t})

    return imm*(output_t[0,...,0]+0.5)
    
def blend_images_horizontally(image1, image2, overlap_width):
    channels = image1.shape[2] if len(image1.shape) == 3 else 1
    # Create weight maps for horizontal blending
    
    weight_map1 = np.tile(np.linspace(1, 0, overlap_width).reshape(1, -1, 1), (image1.shape[0], 1, channels))[:,:,0]
    weight_map2 = np.tile(np.linspace(0, 1, overlap_width).reshape(1, -1, 1), (image1.shape[0], 1, channels))[:,:,0]

    # Extract the overlap regions
    overlap1 = image1[:, -overlap_width:]
    overlap2 = image2[:, :overlap_width]
    
    # Blend the overlap regions
    blended_overlap = (overlap1 * weight_map1 + overlap2 * weight_map2)
    
    # Combine the non-overlapping regions with the blended overlap
    blended_image = np.hstack((image1[:, :-overlap_width], blended_overlap, image2[:, overlap_width:]))
    
    return blended_image

def blend_images_vertically(image1, image2, overlap_height):
    channels = image1.shape[2] if len(image1.shape) == 3 else 1
    # Create weight maps for vertical blending
    weight_map1 = np.tile(np.linspace(1, 0, overlap_height).reshape(-1, 1, 1), (1, image1.shape[1], channels))[:,:,0]
    weight_map2 = np.tile(np.linspace(0, 1, overlap_height).reshape(-1, 1, 1), (1, image1.shape[1], channels))[:,:,0]
    
    # Extract the overlap regions
    overlap1 = image1[-overlap_height:, :]
    overlap2 = image2[:overlap_height, :]
    
    # Blend the overlap regions
    blended_overlap = (overlap1 * weight_map1 + overlap2 * weight_map2)
    
    # Combine the non-overlapping regions with the blended overlap
    blended_image = np.vstack((image1[:-overlap_height, :], blended_overlap, image2[overlap_height:, :]))
    
    return blended_image

def create_mosaic_grid(images, grid_shape, overlap_percentage=10):
    
    # Ensure all images have the same size
    heights = [img.shape[0] for img in images]
    widths = [img.shape[1] for img in images]
    if len(set(heights)) != 1 or len(set(widths)) != 1:
        raise ValueError("All images must have the same dimensions.")
    
    # Calculate the overlap width and height
    overlap_width = int(widths[0] * overlap_percentage / 100)
    overlap_height = int(heights[0] * overlap_percentage / 100)
    
    # Create an empty list to store rows
    rows = []
    
    # Process each row in the grid
    for i in range(grid_shape[0]):
        # Start with the first image in the row
        row_mosaic = images[i * grid_shape[1]]
        
        # Blend each image with the previous one horizontally
        for j in range(1, grid_shape[1]):
            row_mosaic = blend_images_horizontally(row_mosaic, images[i * grid_shape[1] + j], overlap_width)
        
        rows.append(row_mosaic)
    
    # Start with the first row
    mosaic = rows[0]
    
    # Blend each row with the previous one vertically
    for i in range(1, len(rows)):
        mosaic = blend_images_vertically(mosaic, rows[i], overlap_height)
    
    return mosaic.astype('float32')

def extract_number(filename):
    match = re.search(r'tile_(\d+)', filename)
    return int(match.group(1)) if match else -1


def denoise(imarray, size_padded = 244):     
    # Code for denoising immarray, using patch size of size_padded

    patch_size = graph.get_operation_by_name("img").outputs[0].shape[1:]
    padding = int((patch_size[0] - size_padded) / 2)
    
    output_image = np.zeros(imarray.shape)
    imm = imarray.max()
    count = 0
    size_y = 0
    patch_stack = []
    t0 = time.time()
    
    # add padding to image so that x y dimensions are multiples of size_padded + 2 padding:
    target_height = ((imarray.shape[0] + size_padded-1) // size_padded) * size_padded + 2*padding
    target_width = ((imarray.shape[1] + size_padded-1) // size_padded) * size_padded + 2*padding
    rows_to_pad = target_height - imarray.shape[0]
    cols_to_pad = target_width - imarray.shape[1]
    imarray = np.pad(imarray, ((0, rows_to_pad), (0, cols_to_pad)), mode='constant', constant_values=0)
    
    # Divide image into small patches, denoise all patches and store in patch_stack
    
    with tf.compat.v1.Session() as sess:
        for i in range(padding, imarray.shape[0] - padding - size_padded + 1, size_padded):
            size_y = size_y + 1
            for j in range(padding, imarray.shape[1] - padding - size_padded + 1, size_padded):
                patch = imarray[i-padding:i+padding+size_padded, j-padding:j+size_padded+padding]
                
                temp = denoise2D(patch, imm)
                patch_stack.append(temp)
                #im = Image.fromarray(temp[:,:])
                #im.save(os.path.join(outdir, "tile_"+str(count).zfill(2)+".tif")) 
                count = count + 1
    size_x = count / size_y
    t1 = time.time()
    print("denoising finished in ",int(t1-t0), " seconds.")
    
    # stitch patch_stack into single image
    grid_shape = [int(size_y), int(size_x)]
    mosaic = create_mosaic_grid(patch_stack, grid_shape, overlap_percentage=10)
    
    return mosaic


if __name__ == "__main__":
    indir, outdir = argv[1].split(",") 

    if os.path.isdir(outdir):
        print("Found {}".format(outdir))
    else:
        print("Creating {}".format(outdir))
    
        os.makedirs(outdir)

    Image.MAX_IMAGE_PIXELS = None
    GRAPH_PB_PATH = 'Noise2Noise_ML_Model_VM_01.pb'
    with tf.compat.v1.Session() as sess:
        with tf.io.gfile.GFile(GRAPH_PB_PATH,'rb') as f:
            graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())
        sess.graph.as_default()
        tf.compat.v1.import_graph_def(graph_def, name='')


        graph_nodes=[n for n in graph_def.node]


    #Load the graph from the .pb file in TensorFlow v2
    model_path = GRAPH_PB_PATH
    with open(model_path, 'rb') as f:
        graph_def = tf.compat.v1.GraphDef()  # Use tf.compat.v1.GraphDef for compatibility
        graph_def.ParseFromString(f.read())

    # Create a new TensorFlow v2 graph and import the graph_def
    with tf.compat.v1.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name='')



    im = Image.open(indir)
    imarray = np.array(im).astype('float64')
    denoised_imarray = denoise(imarray)
    
    output_name = os.path.basename(indir)
    tiff.imwrite(os.path.join(outdir,output_name + ".tif"), denoised_imarray)

