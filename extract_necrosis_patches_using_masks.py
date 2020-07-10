import os
import sys
import openslide 
import cv2
import math 
import numpy as np 
from xml.etree.ElementTree import parse
import argparse 


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--slide_dir', type=str, default='WSI/',
                       help='You must specify the data directory containing the whole slide images')
    parser.add_argument('--xml_dir', type=str, default='XML/',
                       help='You must specify the data directory containing the whole slide image XML annotations')
    parser.add_argument('--patchmap_dir', type=str, default='PATCHMAPS/',
                       help='You must specify the data directory to store tissue maps')
    parser.add_argument('--tissue_mask_dir', type=str, default='TISSUE_MASK/',
                       help='You must specify the data directory with the stored tissue masks')
    parser.add_argument('--negative_mask_dir', type=str, default='NEGATIVE_MASK/',
                       help='You must specify the data directory with the stored negative masks')
    parser.add_argument('--necrosis_mask_dir', type=str, default='NECROSIS_MASK/',
                       help='You must specify the data directory with the stored necrosis masks')
    parser.add_argument('--necrosis_patches_dir', type=str, default='NECROSIS_PATCHES/',
                       help='You must specify the data directory for the necrosis patches')
    parser.add_argument('--negative_patches_dir', type=str, default='NEGATIVE_PATCHES/',
                       help='You must specify the data directory for the negative patches')
    parser.add_argument('--mask_level', type=int, default= 6,
                       help='The chosen mask level, the highest level is 0')
    parser.add_argument('--level', type=int, default= 1,
                       help='The chosen level for extracting patches, highest magnification is level 0')
    parser.add_argument('--negative_threshold', type=int, default= 0.3,
                       help='Negative threshold between 0 and 1. Negative mask inclusion ratio that select negative patches')
    parser.add_argument('--necrosis_threshold', type=int, default= 0.8,
                       help='Necrosis threshold betwwn 0 and 1. Necrosis mask inclusion ratio that select necrosis patches')
    parser.add_argument('--patch_size', type=int, default= 256,
                       help='Size of patches in pixels')                            

    args = parser.parse_args()

    extract_patches(args)



def extract_patches(args):

    p_size = args.patch_size
    multiply_factor = int(2**args.level)
    step = int(p_size/(2**(args.mask_level-args.level))) # (6 - 2) 

    for root, dirnames, filenames in os.walk(args.slide_dir):
        for file in filenames:

            slide_path = os.path.join(root,file)
            slide_name = slide_path.strip("WSI/")
            slide_name = slide_name.rstrip(".ndpi")

            necrosis_patches_path = str(args.necrosis_patches_dir) + slide_name + '/'
            if not os.path.exists(necrosis_patches_path):
                os.makedirs(necrosis_patches_path)

            negative_patches_path = str(args.negative_patches_dir) + slide_name + '/'
            if not os.path.exists(negative_patches_path):
                os.makedirs(negative_patches_path)
       
            slide = openslide.OpenSlide(slide_path)
            slide_map = cv2.imread(args.patchmap_dir + slide_name + '_map.png',-1)

            necrosis_mask = cv2.imread(args.necrosis_mask_dir + slide_name + '_necrosis_mask.png' , 0)
            negative_mask = cv2.imread(args.negative_mask_dir + slide_name + '_negative_mask.png' , 0)

            width, height = np.array(slide.level_dimensions[args.level])//p_size # lowest level is 0 but use 1 for 40x magnification, 2 for 20x etc.

            for i in range(width):
                for j in range(height):
                    necrosis_mask_sum = necrosis_mask[step * j : step * (j+1), step * i : step * (i+1)].sum()
                    negative_mask_sum = negative_mask[step * j : step * (j+1), step * i : step * (i+1)].sum()

                    mask_max = step * step * 255
                
                    necrosis_area_ratio = necrosis_mask_sum / mask_max
                    negative_area_ratio = negative_mask_sum / mask_max

                    # extract necrosis patch
                    if (necrosis_area_ratio > args.necrosis_threshold):
                        patch_name = necrosis_patches_path + 'nec_' + str(slide_name) + '_' + str(args.level) + '_w_' + str(i) + '_h_' + str(j) + '_.png'
                        patch = slide.read_region(((p_size*i*multiply_factor),(p_size*j*multiply_factor)), args.level, (p_size, p_size)) # takes coordinates on top left corner at level 0 always therefore resize coordinates 
                        
                        # divide by difference of mag levels i.e 2^2 
                        #patch.save(patch_name)
                        cv2.rectangle(slide_map, (step*i,step*j), (step*(i+1),step*(j+1)), (0,0,255), 1)
                    
                    # extract negative patch
                    if (negative_area_ratio > args.negative_threshold) and (necrosis_area_ratio == 0):
                        patch_name = negative_patches_path + 'nega_' + str(slide_name) + '_' + str(args.level) + '_w_' + str(i) + '_h_' + str(j) + '_.png'
                        patch = slide.read_region(((p_size*i*multiply_factor),(p_size*j*multiply_factor)), args.level, (p_size, p_size)) # takes coordinates on top left corner at level 0 always therefore resize coordinates 
                        
                        # divide by difference of mag levels i.e 2^2 
                        #patch.save(patch_name)
                        cv2.rectangle(slide_map, (step*i,step*j), (step*(i+1),step*(j+1)), (0,225,255), 1)

                    # nothing
                    else:
                        pass

            cv2.imwrite(args.patchmap_dir + '/' + slide_name + '_map.png', slide_map)



if __name__ == '__main__':
    main()