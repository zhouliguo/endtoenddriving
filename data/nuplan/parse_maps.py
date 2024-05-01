# https://geopandas.org/en/stable/docs.html

import geopandas as gpd
from matplotlib import pyplot as plt
import pyogrio
import os
from shapely import geometry
from typing import Dict, List, Tuple, Union
import numpy.typing as npt
import numpy as np
map1 = gpd.read_file('data/nuplan/maps/sg-one-north/9.17.1964/map.gpkg')
map2 = gpd.read_file('data/nuplan/maps/us-ma-boston/9.12.1817/map.gpkg')
path_on_disk = os.path.join('data/nuplan/maps/us-nv-las-vegas-strip/9.15.1915/map.gpkg')
map3 = gpd.read_file(path_on_disk,layer="meta", engine="pyogrio")

def load_vector_layer(location: str, layer_name: str)-> gpd.geodataframe:
    # The projected coordinate system depends on which UTM zone the mapped location is in.
    map = gpd.read_file(path_on_disk,layer="meta", engine="pyogrio")
    projection_system = map[map["key"] == "projectedCoordSystem"]["value"].iloc[0]
    gdf_in_pixel_coords = pyogrio.read_dataframe(path_on_disk, layer= layer_name, fid_as_index=True)
    gdf_in_utm_coords = gdf_in_pixel_coords.to_crs(projection_system)
    # For backwards compatibility, cast the index to string datatype.
    #  and mirror it to the "fid" column.
    gdf_in_utm_coords.index = gdf_in_utm_coords.index.map(str)
    gdf_in_utm_coords["fid"] = gdf_in_utm_coords.index
    return gdf_in_utm_coords

def build_lane_segments_from_blps(
    candidate_blps: gpd.geodataframe,
    ls_coords: List[List[List[float]]],
    ls_conns: List[List[int]],
    ls_groupings: List[List[int]],
    cross_blp_conns: Dict[str, List[int]],
) -> None:
    """
    Process candidate baseline paths to small portions of lane-segments with connection info recorded.
    :param candidate_blps: Candidate baseline paths to be cut to lane_segments
    :param ls_coords: Output data recording lane-segment coordinates in format of [N, 2, 2]
    :param ls_conns: Output data recording lane-segment connection relations in format of [M, 2]
    :param ls_groupings: Output data recording lane-segment indices associated with each lane in format
        [num_lanes, num_segments_in_lane]
    :param: cross_blp_conns: Output data recording start_idx/end_idx for each baseline path with id as key.
    """
    for _, blp in candidate_blps.iterrows():
        blp_id = blp['fid']
        px, py = blp.geometry.coords.xy
        ls_num = len(px) - 1
        blp_start_ls = len(ls_coords)
        blp_end_ls = blp_start_ls + ls_num - 1
        ls_grouping = []
        for idx in range(ls_num):
            curr_pt, next_pt = [px[idx], py[idx]], [px[idx + 1], py[idx + 1]]
            ls_idx = len(ls_coords)
            if idx > 0:
                ls_conns.append([ls_idx - 1, ls_idx])
            ls_coords.append([curr_pt, next_pt])
            ls_grouping.append(ls_idx)
        ls_groupings.append(ls_grouping)
        cross_blp_conns[blp_id] = [blp_start_ls, blp_end_ls]

def connect_blp_predecessor(
    blp_id: str, lane_conn_info: gpd.geodataframe, cross_blp_conns: Dict[str, List[int]], ls_conns: List[List[int]]
) -> None:
    """
    Given a specific baseline path id, find its predecessor and update info in ls_connections information.
    :param blp_id: a specific baseline path id to query
    :param lane_conn_info: baseline paths information in intersections contains the from_blp/to_blp info
    :param cross_blp_conns: Dict to record the baseline path id as key(str) and [blp_start_ls_idx, blp_end_ls_idx] pair
        as value (List[int])
    :param ls_conns: lane_segment_connection to record the [from_ls_idx, to_ls_idx] connection info, updated with
        predecessors found.
    """
    blp_start, blp_end = cross_blp_conns[blp_id]
    predecessor_blp = lane_conn_info[lane_conn_info['to_blp'] == blp_id]
    predecessor_list = predecessor_blp['fid'].to_list()

    for predecessor_id in predecessor_list:
        predecessor_start, predecessor_end = cross_blp_conns[predecessor_id]
        ls_conns.append([predecessor_end, blp_start])

def connect_blp_successor(
    blp_id: str, lane_conn_info: gpd.geodataframe, cross_blp_conns: Dict[str, List[int]], ls_conns: List[List[int]]
) -> None:
    """
    Given a specific baseline path id, find its successor and update info in ls_connections information.
    :param blp_id: a specific baseline path id to query
    :param lane_conn_info: baseline paths information in intersections contains the from_blp/to_blp info
    :param cross_blp_conns: Dict to record the baseline path id as key(str) and [blp_start_ls_idx, blp_end_ls_idx] pair
        as value (List[int])
    :param ls_conns: lane_segment_connnection to record the [from_ls_idx, to_ls_idx] connection info, updated with
        predecessors found.
    """
    blp_start, blp_end = cross_blp_conns[blp_id]
    successor_blp = lane_conn_info[lane_conn_info['from_blp'] == blp_id]
    successor_list = successor_blp['fid'].to_list()

    for successor_id in successor_list:
        successor_start, successor_end = cross_blp_conns[successor_id]
        ls_conns.append([blp_end, successor_start])

def plot_map(ego_x:float,ego_y:float):
    # load the map information and project the latitude and longitude coordinates to UTM coordinates
    blps_gdf = load_vector_layer(path_on_disk, 'baseline_paths')  # type: gpd.geodataframe
    lane_poly_gdf = load_vector_layer(path_on_disk, 'lanes_polygons')  
    intersections_gdf = load_vector_layer(path_on_disk, 'intersections')  
    lane_connectors_gdf = load_vector_layer(path_on_disk, 'lane_connectors') 
    lane_groups_gdf = load_vector_layer(path_on_disk, 'lane_groups_polygons')  

    # data enhancement
    blps_in_lanes = blps_gdf[blps_gdf['lane_fid'].notna()]
    blps_in_intersections = blps_gdf[blps_gdf['lane_connector_fid'].notna()]
    # enhance blps_in_lanes
    lane_group_info = lane_poly_gdf[['lane_fid', 'lane_group_fid']]
    blps_in_lanes = blps_in_lanes.merge(lane_group_info, on='lane_fid', how='outer')

    # enhance blps_in_intersections
    lane_connectors_gdf['lane_connector_fid'] = lane_connectors_gdf['fid']
    lane_conns_info = lane_connectors_gdf[
        ['lane_connector_fid', 'intersection_fid', 'exit_lane_fid', 'entry_lane_fid']
    ]
    # Convert the exit_fid field of both data frames to the same dtype for merging.
    lane_conns_info = lane_conns_info.astype({'lane_connector_fid': int})
    blps_in_intersections = blps_in_intersections.astype({'lane_connector_fid': int})
    blps_in_intersections = blps_in_intersections.merge(lane_conns_info, on='lane_connector_fid', how='outer')

    #determine the range of the map we want to crop using ego point as the circle center
    xrange = [-60, 60]
    yrange = [-60, 60]
    x_min, x_max = ego_x + xrange[0], ego_x + xrange[1]
    y_min, y_max = ego_y + yrange[0], ego_y + yrange[1]

    patch = geometry.box(x_min, y_min, x_max, y_max)
    candidate_lane_groups = lane_groups_gdf[lane_groups_gdf["geometry"].intersects(patch)]#datatype:gpd.geodataframe

    # Select in-range blps
    candidate_intersections = intersections_gdf[intersections_gdf["geometry"].intersects(patch)]
    candidate_blps_in_lanes = blps_in_lanes[
        blps_in_lanes['lane_group_fid'].isin(candidate_lane_groups['fid'].astype(int))
    ]
    candidate_blps_in_intersections = blps_in_intersections[
        blps_in_intersections['intersection_fid'].isin(candidate_intersections['fid'].astype(int))
    ]

    return candidate_blps_in_lanes,candidate_blps_in_intersections,candidate_intersections
    


# ls_coordinates_list: List[List[List[float]]] = []
# ls_connections_list: List[List[int]] = []
# ls_groupings_list: List[List[int]] = []
# cross_blp_connection: Dict[str, List[int]] = dict()


# build_lane_segments_from_blps(
#     candidate_blps_in_lanes, ls_coordinates_list, ls_connections_list, ls_groupings_list, cross_blp_connection
# )
# # generate lane_segments from blps in intersections
# build_lane_segments_from_blps(
#     candidate_blps_in_intersections,
#     ls_coordinates_list,
#     ls_connections_list,
#     ls_groupings_list,
#     cross_blp_connection,
# )

# generate connections between blps
#for blp_id, blp_info in cross_blp_connection.items():
    # Add predecessors
    #connect_blp_predecessor(blp_id, candidate_blps_in_intersections, cross_blp_connection, ls_connections_list)
    # Add successors
    #connect_blp_successor(blp_id, candidate_blps_in_intersections, cross_blp_connection, ls_connections_list)

# ls_coordinates: npt.NDArray[np.float64] = np.asarray(ls_coordinates_list, self.translation_np.dtype)
# ls_connections: npt.NDArray[np.int64] = np.asarray(ls_connections_list, np.int64)
# Transform the lane coordinates from global frame to ego vehicle frame.
# Flatten ls_coordinates from (num_ls, 2, 2) to (num_ls * 2, 2) for easier processing.
# ls_coordinates = ls_coordinates.reshape(-1, 2)
# ls_coordinates = ls_coordinates - self.translation_np[:2]
# ls_coordinates = self.rotate_2d_points2d_to_ego_vehicle_frame(ls_coordinates)
# ls_coordinates = ls_coordinates.reshape(-1, 2, 2).astype(np.float32)







#print(blps_in_lanes)

#map2.plot()
#plt.show()
