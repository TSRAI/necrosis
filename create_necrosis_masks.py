# TS RAI 2020 

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
                       help='You must specify the data directory to store WSI maps')
    parser.add_argument('--copymap_dir', type=str, default='COPYMAPS/',
                       help='You must specify the data directory to store WSI maps')
    parser.add_argument('--map_dir', type=str, default='MAPS/',
                       help='You must specify the data directory to store WSI maps')
    parser.add_argument('--tissue_mask_dir', type=str, default='TISSUE_MASK/',
                       help='You must specify the data directory to store tissue masks')
    parser.add_argument('--negative_mask_dir', type=str, default='NEGATIVE_MASK/',
                       help='You must specify the data directory to store negative masks')
    parser.add_argument('--necrosis_mask_dir', type=str, default='NECROSIS_MASK/',
                       help='You must specify the data directory to store necrosis masks')
    parser.add_argument('--mask_level', type=int, default= 6,
                       help='The chosen mask level, the highest level is 0')

    args = parser.parse_args()

    create_masks(args)



def read_xml(slide_name, xml_direc, mask_level):
    ''' read xml files which has class coordinates list
        return coordinates of class areas
    Args:
        slide_num (int): number of slide used
        mask_level (int): level of mask
    '''

    slide_name = slide_name.rstrip(".ndpi")
    slide_name = slide_name.strip("WSI/")
    path = xml_direc + slide_name + ".xml"

    xml = parse(path).getroot()

    coors_list = []
    coors = []
    for areas in xml.iter('Coordinates'):
        for area in areas:
            coors.append([round(float(area.get('X'))/(2**mask_level)),
                            round(float(area.get('Y'))/(2**mask_level))])
        coors_list.append(coors)
        coors=[]

    return coors_list



def make_mask(slide_obj, xml_direc, mask_level, map_dir, patchmap_dir, copymap_dir, tissue_mask_dir, necrosis_mask_dir, negative_mask_dir):

    slide_name = slide_obj.rstrip(".ndpi")
    slide_name = slide_name.strip("WSI/")

     # create directories if they don't exist 
    if not os.path.exists(map_dir):
        os.makedirs(map_dir)
    if not os.path.exists(patchmap_dir):
        os.makedirs(patchmap_dir)
    if not os.path.exists(copymap_dir):
        os.makedirs(copymap_dir)
    if not os.path.exists(tissue_mask_dir):
        os.makedirs(tissue_mask_dir)
    if not os.path.exists(necrosis_mask_dir):
        os.makedirs(necrosis_mask_dir)
    if not os.path.exists(negative_mask_dir):
        os.makedirs(negative_mask_dir)


    # directories 
    map_path = map_dir + str(slide_name) + '_map.png'
    patchmap_path = patchmap_dir + str(slide_name) + '_map.png'
    copymap_path = copymap_dir + str(slide_name) + '_map.png'
    necrosis_mask_path = necrosis_mask_dir + str(slide_name) + '_necrosis_mask.png'
    negative_mask_path = negative_mask_dir + str(slide_name) + '_negative_mask.png'
    tissue_mask_path = tissue_mask_dir + str(slide_name) + '_tissue_mask.png'
    
    # load slide
    slide = openslide.OpenSlide(slide_obj)
    slide_map = np.array(slide.get_thumbnail(slide.level_dimensions[mask_level]))

    # load xml coords
    coords = read_xml(slide_obj, xml_direc, mask_level)
    

    # draw boundary of tumor in map
    for coors in coords:
        cv2.drawContours(slide_map, np.array([coors]), -1, (0,255,0), 3)
    RGB_img = cv2.cvtColor(slide_map, cv2.COLOR_BGR2RGB) # OpenCV uses BGR as its default colour order for images, matplotlib uses RGB
    cv2.imwrite(map_path, RGB_img)
    cv2.imwrite(copymap_path, RGB_img)
    cv2.imwrite(patchmap_path, RGB_img)

    necrosis_mask = np.zeros(slide.level_dimensions[mask_level][::-1])

    for coors in coords:
        cv2.drawContours(necrosis_mask, np.array([coors]), -1, 255, -1)
    cv2.imwrite(necrosis_mask_path, necrosis_mask)

    slide_lv = slide.read_region((0, 0), mask_level, slide.level_dimensions[mask_level])
    slide_lv = cv2.cvtColor(np.array(slide_lv), cv2.COLOR_RGBA2RGB)
    slide_lv = cv2.cvtColor(slide_lv, cv2.COLOR_BGR2HSV)
    slide_lv = slide_lv[:, :, 1]
    _,tissue_mask = cv2.threshold(slide_lv, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    cv2.imwrite(tissue_mask_path, np.array(tissue_mask))

    necrosis_mask = cv2.imread(necrosis_mask_path, 0) 
    height, width = np.array(necrosis_mask).shape
    for i in range(width):
        for j in range(height):
            if necrosis_mask[j][i] > 127:
                tissue_mask[j][i] = 0
    negative_mask = np.array(tissue_mask)
    cv2.imwrite(negative_mask_path, negative_mask)


def create_masks(args):
    for root, dirnames, filenames in os.walk(args.slide_dir):
        for file in filenames:
            slide = os.path.join(root,file)
            print(slide)
            make_mask(slide, args.xml_dir, args.mask_level, args.map_dir, args.patchmap_dir, args.copymap_dir, args.tissue_mask_dir, args.necrosis_mask_dir, args.negative_mask_dir)


if __name__ == '__main__':
    main()