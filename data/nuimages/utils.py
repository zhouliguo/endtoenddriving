# nuScenes dev-kit.
# Code written by Asha Asvathaman & Holger Caesar, 2020.

import base64
import os
from typing import List, Dict
import warnings

import matplotlib.font_manager
from PIL import ImageFont
import numpy as np
from pycocotools import mask as cocomask

from typing import Dict, Tuple, List, Optional

def category_to_detection_name(category_name: str) -> Optional[str]:
    """
    Default label mapping from nuScenes to nuScenes detection classes.
    Note that pedestrian does not include personal_mobility, stroller and wheelchair.
    :param category_name: Generic nuScenes class.
    :return: nuScenes detection class.
    """
    detection_mapping = {
        'movable_object.barrier': 'barrier',
        'vehicle.bicycle': 'bicycle',
        'vehicle.bus.bendy': 'bus',
        'vehicle.bus.rigid': 'bus',
        'vehicle.car': 'car',
        'vehicle.construction': 'construction_vehicle',
        'vehicle.motorcycle': 'motorcycle',
        'human.pedestrian.adult': 'pedestrian',
        'human.pedestrian.child': 'pedestrian',
        'human.pedestrian.construction_worker': 'pedestrian',
        'human.pedestrian.police_officer': 'pedestrian',
        'movable_object.trafficcone': 'traffic_cone',
        'vehicle.trailer': 'trailer',
        'vehicle.truck': 'truck'
    }

    if category_name in detection_mapping:
        return detection_mapping[category_name]
    else:
        return None
    
def get_colormap() -> Dict[str, Tuple[int, int, int]]:
    """
    Get the defined colormap.
    :return: A mapping from the class names to the respective RGB values.
    """

    classname_to_color = {  # RGB.
        "noise": (0, 0, 0),  # Black.
        "animal": (70, 130, 180),  # Steelblue
        "human.pedestrian.adult": (0, 0, 230),  # Blue
        "human.pedestrian.child": (135, 206, 235),  # Skyblue,
        "human.pedestrian.construction_worker": (100, 149, 237),  # Cornflowerblue
        "human.pedestrian.personal_mobility": (219, 112, 147),  # Palevioletred
        "human.pedestrian.police_officer": (0, 0, 128),  # Navy,
        "human.pedestrian.stroller": (240, 128, 128),  # Lightcoral
        "human.pedestrian.wheelchair": (138, 43, 226),  # Blueviolet
        "movable_object.barrier": (112, 128, 144),  # Slategrey
        "movable_object.debris": (210, 105, 30),  # Chocolate
        "movable_object.pushable_pullable": (105, 105, 105),  # Dimgrey
        "movable_object.trafficcone": (47, 79, 79),  # Darkslategrey
        "static_object.bicycle_rack": (188, 143, 143),  # Rosybrown
        "vehicle.bicycle": (220, 20, 60),  # Crimson
        "vehicle.bus.bendy": (255, 127, 80),  # Coral
        "vehicle.bus.rigid": (255, 69, 0),  # Orangered
        "vehicle.car": (255, 158, 0),  # Orange
        "vehicle.construction": (233, 150, 70),  # Darksalmon
        "vehicle.emergency.ambulance": (255, 83, 0),
        "vehicle.emergency.police": (255, 215, 0),  # Gold
        "vehicle.motorcycle": (255, 61, 99),  # Red
        "vehicle.trailer": (255, 140, 0),  # Darkorange
        "vehicle.truck": (255, 99, 71),  # Tomato
        "flat.driveable_surface": (0, 207, 191),  # nuTonomy green
        "flat.other": (175, 0, 75),
        "flat.sidewalk": (75, 0, 75),
        "flat.terrain": (112, 180, 60),
        "static.manmade": (222, 184, 135),  # Burlywood
        "static.other": (255, 228, 196),  # Bisque
        "static.vegetation": (0, 175, 0),  # Green
        "vehicle.ego": (255, 240, 245)
    }

    return classname_to_color


def annotation_name(attributes: List[dict],
                    category_name: str,
                    with_attributes: bool = False) -> str:
    """
    Returns the "name" of an annotation, optionally including the attributes.
    :param attributes: The attribute dictionary.
    :param category_name: Name of the object category.
    :param with_attributes: Whether to print the attributes alongside the category name.
    :return: A human readable string describing the annotation.
    """
    outstr = category_name

    if with_attributes:
        atts = [attribute['name'] for attribute in attributes]
        if len(atts) > 0:
            outstr = outstr + "--" + '.'.join(atts)

    return outstr


def mask_decode(mask: dict) -> np.ndarray:
    """
    Decode the mask from base64 string to binary string, then feed it to the external pycocotools library to get a mask.
    :param mask: The mask dictionary with fields `size` and `counts`.
    :return: A numpy array representing the binary mask for this class.
    """
    # Note that it is essential to copy the mask here. If we use the same variable we will overwrite the NuImage class
    # and cause the Jupyter Notebook to crash on some systems.
    new_mask = mask.copy()
    new_mask['counts'] = base64.b64decode(mask['counts'])
    return cocomask.decode(new_mask)


def get_font(fonts_valid: List[str] = None, font_size: int = 15) -> ImageFont:
    """
    Check if there is a desired font present in the user's system. If there is, use that font; otherwise, use a default
    font.
    :param fonts_valid: A list of fonts which are desirable.
    :param font_size: The size of the font to set. Note that if the default font is used, then the font size
        cannot be set.
    :return: An ImageFont object to use as the font in a PIL image.
    """
    # If there are no desired fonts supplied, use a hardcoded list of fonts which are desirable.
    if fonts_valid is None:
        fonts_valid = ['FreeSerif.ttf', 'FreeSans.ttf', 'Century.ttf', 'Calibri.ttf', 'arial.ttf']

    # Find a list of fonts within the user's system.
    fonts_in_sys = matplotlib.font_manager.findSystemFonts(fontpaths=None, fontext='ttf')
    # Sort the list of fonts to ensure that the desired fonts are always found in the same order.
    fonts_in_sys = sorted(fonts_in_sys)
    # Of all the fonts found in the user's system, check if any of them are desired.
    for font_in_sys in fonts_in_sys:
        if any(os.path.basename(font_in_sys) in s for s in fonts_valid):
            return ImageFont.truetype(font_in_sys, font_size)

    # If none of the fonts in the user's system are desirable, then use the default font.
    warnings.warn('No suitable fonts were found in your system. '
                  'A default font will be used instead (the font size will not be adjustable).')
    return ImageFont.load_default()


def name_to_index_mapping(category: List[dict]) -> Dict[str, int]:
    """
    Build a mapping from name to index to look up index in O(1) time.
    :param category: The nuImages category table.
    :return: The mapping from category name to category index.
    """
    # The 0 index is reserved for non-labelled background; thus, the categories should start from index 1.
    # Also, sort the categories before looping so that the order is always the same (alphabetical).
    name_to_index = dict()
    i = 1
    sorted_category: List = sorted(category.copy(), key=lambda k: k['name'])
    for c in sorted_category:
        # Ignore the vehicle.ego and flat.driveable_surface classes first; they will be mapped later.
        if c['name'] != 'vehicle.ego' and c['name'] != 'flat.driveable_surface':
            name_to_index[c['name']] = i
            i += 1

    assert max(name_to_index.values()) < 24, \
        'Error: There are {} classes (excluding vehicle.ego and flat.driveable_surface), ' \
        'but there should be 23. Please check your category.json'.format(max(name_to_index.values()))

    # Now map the vehicle.ego and flat.driveable_surface classes.
    name_to_index['flat.driveable_surface'] = 24
    name_to_index['vehicle.ego'] = 31

    # Ensure that each class name is uniquely paired with a class index, and vice versa.
    assert len(name_to_index) == len(set(name_to_index.values())), \
        'Error: There are {} class names but {} class indices'.format(len(name_to_index),
                                                                      len(set(name_to_index.values())))

    return name_to_index