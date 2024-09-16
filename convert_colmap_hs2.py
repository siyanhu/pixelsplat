import root_file_io as fio

from PIL import Image
import io
import re
import copy
import numpy as np
import subprocess
from collections import defaultdict
from colorama import Fore
from datetime import datetime

from typing import Literal, TypedDict
from jaxtyping import Float, Int, UInt8
import torch
from torch import Tensor
import json

from ruamel.yaml import YAML


TARGET_BYTES_PER_CHUNK = int(1e8)


class Metadata(TypedDict):
    url: str
    timestamps: Int[Tensor, "camera"]
    cameras: Float[Tensor, "camera entry"]


class Example(Metadata):
    key: str
    images: list[UInt8[Tensor, "..."]]


def cyan(text: str) -> str:
    return f"{Fore.CYAN}{text}{Fore.RESET}"


def quaternion_to_rotation_matrix(qw, qx, qy, qz):
    """Convert a quaternion to a rotation matrix."""
    R = np.array([
        [1 - 2*qy**2 - 2*qz**2,     2*qx*qy - 2*qz*qw,     2*qx*qz + 2*qy*qw],
        [    2*qx*qy + 2*qz*qw, 1 - 2*qx**2 - 2*qz**2,     2*qy*qz - 2*qx*qw],
        [    2*qx*qz - 2*qy*qw,     2*qy*qz + 2*qx*qw, 1 - 2*qx**2 - 2*qy**2]
    ])
    return R


def colmap_intrinsics_to_camera_matrix(camera_model, params):
    """
    Convert COLMAP intrinsics format to a 3x3 camera matrix.
    
    Args:
    camera_model: String representing the COLMAP camera model
    params: List of parameters specific to the camera model
    
    Returns:
    3x3 camera matrix
    """
    if camera_model == "SIMPLE_PINHOLE":
        f, cx, cy = params
        return np.array([
            [f, 0, cx],
            [0, f, cy],
            [0, 0, 1]
        ])
    
    elif camera_model == "PINHOLE":
        fx, fy, cx, cy = params
        return np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ])
    
    elif camera_model == "SIMPLE_RADIAL":
        f, cx, cy, k = params
        # Note: The radial distortion parameter k is not used in the camera matrix
        return np.array([
            [f, 0, cx],
            [0, f, cy],
            [0, 0, 1]
        ])
    
    else:
        raise ValueError(f"Unsupported camera model: {camera_model}")
    

def colmap_extrinsics_to_transformation_matrix(qw, qx, qy, qz, tx, ty, tz, scale_factor=1):
    """
    Convert COLMAP extrinsics format to a 4x4 transformation matrix.
    
    Args:
    qw, qx, qy, qz: Quaternion components
    tx, ty, tz: Translation components
    
    Returns:
    4x4 transformation matrix
    """
    # Convert quaternion to rotation matrix
    R = quaternion_to_rotation_matrix(qw, qx, qy, qz)
    
    # Create translation vector
    t = np.array([[tx], [ty], [tz]])

    R, t = normalize_extrinsics(R, t, scale_factor)
    
    # Construct 4x4 transformation matrix
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t.flatten()
    
    return T


def get_colmap_extrinsic(extri_file):
    """
    Taken from https://github.com/colmap/colmap/blob/dev/scripts/python/read_write_model.py
    """
    camera_positions = []
    with open(extri_file, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                image_id = int(elems[0])
                qvec = np.array(tuple(map(float, elems[1:5])))
                tvec = np.array(tuple(map(float, elems[5:8])))
                camera_id = int(elems[8])
                image_name = elems[9]
                elems = fid.readline().split()
                xys = np.column_stack([tuple(map(float, elems[0::3])),
                                       tuple(map(float, elems[1::3]))])
                point3D_ids = np.array(tuple(map(int, elems[2::3])))
                position = np.array([tvec[0], tvec[1], tvec[2]])
                camera_positions.append(position)
    
    camera_positions_np = np.array(camera_positions)
    distances = np.linalg.norm(camera_positions_np[:, None] - camera_positions_np, axis=2)
    scale_factor = np.median(distances[distances > 0])

    images = {}
    with open(extri_file, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                image_id = int(elems[0])
                qvec = np.array(tuple(map(float, elems[1:5])))
                tvec = np.array(tuple(map(float, elems[5:8])))
                camera_id = int(elems[8])
                image_name = elems[9]
                elems = fid.readline().split()
                xys = np.column_stack([tuple(map(float, elems[0::3])),
                                       tuple(map(float, elems[1::3]))])
                point3D_ids = np.array(tuple(map(int, elems[2::3])))

                images[image_name] = {
                    "image_id": image_id,
                    "qvec": qvec,
                    "tvec": tvec,
                    "extrinsic": colmap_extrinsics_to_transformation_matrix(
                        qvec[0], 
                        qvec[1],
                        qvec[2],
                        qvec[3],
                        tvec[0],
                        tvec[1],
                        tvec[2],
                        # scale_factor=scale_factor
                        ),
                    "camera_id": camera_id,
                    "image_name": image_name,
                    "xys": xys,
                    "point3d_ids": point3D_ids
                }
    return images


def get_colmap_intrinsic(intri_file):
    """
    Taken from https://github.com/colmap/colmap/blob/dev/scripts/python/read_write_model.py
    """
    cameras = {}
    with open(intri_file, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                camera_id = int(elems[0])

                if camera_id in cameras:
                    # print(len(cameras))
                    continue

                model = elems[1]
                # assert model == "PINHOLE", "While the loader support other types, the rest of the code assumes PINHOLE"
                width = int(elems[2])
                height = int(elems[3])
                params = np.array(tuple(map(float, elems[4:])))

                new_width = 360
                new_height = 640
                scale_x = new_width / width
                scale_y = new_height / height

                f, cx, cy = params
                new_f = f * (scale_x + scale_y) / 2
                new_cx = cx * scale_x
                new_cy = cy * scale_y
                new_params = [new_f, new_cx, new_cy]
                params = np.array(tuple(map(float, new_params)))
                # params = np.array(tuple(map(float, elems[4:])))

                init_intri = colmap_intrinsics_to_camera_matrix(model, params)
                normalised_intri = normalize_intrinsics(init_intri, width, height)
                cameras[camera_id] = {
                    "camera_id": camera_id,
                    "model": model,
                    # "width": new_width,
                    # "height": new_height,
                    "width": width,
                    "height": height,
                    "params": params,
                    # "intrinsic": normalised_intri
                    "intrinsic": init_intri
                }
    return cameras


def normalize_intrinsics(K, width, height):
    K_norm = K.copy()
    K_norm[0, 0] /= width  # fx
    K_norm[1, 1] /= width  # fy
    K_norm[0, 2] /= width  # cx
    K_norm[1, 2] /= height  # cy
    return K_norm


def normalize_extrinsics(R, t, scale_factor):
    t_norm = t / scale_factor
    return R, t_norm


def get_colmap_points3d(pnt3d_file):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadPoints3DText(const std::string& path)
        void Reconstruction::WritePoints3DText(const std::string& path)
    """
    points3D = {}
    image_points = defaultdict(list)
    with open(pnt3d_file, 'r') as f:
        for line in f:
            if line[0] != '#':
                data = line.split()
                point3D_id = int(data[0])
                x, y, z = map(float, data[1:4])
                r, g, b = map(int, data[4:7])
                error = float(data[7])
                image_ids = list(map(int, data[8::2]))
                point2D_idxs = list(map(int, data[9::2]))
                points3D[point3D_id] = {'xyz': [x, y, z], 'rgb': [r, g, b], 'error': error, 'image_ids': image_ids, 'point2D_idxs': point2D_idxs}
    return points3D


def build_camera_info(intr, extr):
    downSample = 1.0
    scale_factor = 1.0 / 20

    intr[:2] *= 4
    intr[:2] = intr[:2] * downSample
    extr[:3, 3] *= scale_factor

    return intr, extr


def get_params(original_image_path, intrinsicss, extrinsicss, points3D={}, padding_factor=1):
    (filedir, file_name, fileext) = fio.get_filename_components(original_image_path)
    (nextfiledir, next_seq_name, next_fileext) = fio.get_filename_components(filedir)
    image_name = fio.sep.join([next_seq_name, file_name]) + '.' + fileext

    if image_name not in extrinsicss:
        print("[ERROR] Cannot find image in extrinsics", original_image_path, image_name)
        return
    
    single_extrisic = extrinsicss[image_name]
    single_intrinsic = copy.deepcopy(intrinsicss[single_extrisic['camera_id']])

    qvec = extrinsicss[image_name]['qvec']
    R = quaternion_to_rotation_matrix(
        qvec[0], 
        qvec[1],
        qvec[2],
        qvec[3])
    
    t = extrinsicss[image_name]['tvec']

    near = 1.0
    far = 100.0
    # if len(points3D) > 1:
    #     visible_points = [points3D[point3D_id]['xyz'] for point3D_id in points3D if extrinsicss[image_name]['image_id'] in points3D[point3D_id]['image_ids']]
    #     visible_points_array = np.array(visible_points)
    #     points_cam = np.dot(R, visible_points_array.T).T + t
    #     depths = points_cam[:, 2]
    #     min_depth = np.min(depths)
    #     max_depth = np.max(depths)
    #     near = max(min_depth / padding_factor, 0.1)
    #     far = max_depth * padding_factor

    intri, extri = build_camera_info(single_intrinsic['intrinsic'], single_extrisic['extrinsic'])
    return original_image_path, image_name, intri, extri, near, far


def load_metadata(intrinsics, world2cams) -> Metadata:
    timestamps = []
    cameras = []
    url = ""

    for vid, intr in intrinsics.items():
        timestamps.append(int(vid))
        # normalized the intr
        fx = intr[0, 0]
        fy = intr[1, 1]
        cx = intr[0, 2]
        cy = intr[1, 2]
        w = 2.0 * cx
        h = 2.0 * cy
        saved_fx = fx / w
        saved_fy = fy / h
        saved_cx = 0.5
        saved_cy = 0.5
        camera = [saved_fx, saved_fy, saved_cx, saved_cy, 0.0, 0.0]

        w2c = world2cams[vid]
        camera.extend(w2c[:3].flatten().tolist())
        cameras.append(np.array(camera))

    timestamps = torch.tensor(timestamps, dtype=torch.int64)
    cameras = torch.tensor(np.stack(cameras), dtype=torch.float32)

    return {
        "url": url,
        "timestamps": timestamps,
        "cameras": cameras,
    }


def load_raw(path) -> UInt8[Tensor, " length"]:
    return torch.tensor(np.memmap(path, dtype="uint8", mode="r"))


def get_size(path) -> int:
    """Get file or folder size in bytes."""
    return int(subprocess.check_output(["du", "-b", path]).split()[0].decode("utf-8"))


def load_raw_resize(path, save_path) -> UInt8[Tensor, " length"]:
    image = Image.open(path)
    resized_image = image.resize((640, 360))
    byte_arr = io.BytesIO()
    resized_image.save(byte_arr, format='PNG')
    resized_image.save(save_path)
    byte_arr = byte_arr.getvalue()
    memmap_array = np.frombuffer(byte_arr, dtype=np.uint8)
    # Load the resized image into a PyTorch tensor
    tensor = torch.tensor(memmap_array)
    file_size = len(byte_arr)
    return file_size, tensor


def calculate_file_sizes(file_paths):
    total_size = 0
    errors = []

    for path in file_paths:
        try:
            size = int(subprocess.check_output(["du", "-b", path]).split()[0].decode("utf-8"))
            total_size += size
        except FileNotFoundError:
            errors.append(f"{path}: File not found")
        except OSError as e:
            errors.append(f"{path}: Error: {str(e)}")

    return total_size, errors


def parse_pairs_file(filename, data_dir, n=-1):
    pairs_path_dict = {}
    pairs_label_dict = {}
    
    with open(filename, 'r') as file:
        for line in file:
            # Split the line into key and value
            key, value = line.strip().split()
            key_path = fio.createPath(fio.sep, [data_dir, 'test_full_byorder_59', 'images'], key)
            value_path = fio.createPath(fio.sep, [data_dir, 'train_full_byorder_85', 'images'], value)
            
            if (fio.file_exist(key_path) == False) or (fio.file_exist(value_path) == False):
                continue
            if key_path not in pairs_path_dict:
                pairs_path_dict[key] = key_path
            if value_path not in pairs_path_dict:
                pairs_path_dict[value] = value_path

            if key not in pairs_label_dict:
                pairs_label_dict[key] = []
            # elif len(pairs_label_dict[key]) >= n:
            #     continue
            pairs_label_dict[key].append(value)

    pairs_label_result = {}
    for key, value in pairs_label_dict.items():
        # new_value = [value[0], value[-1]]

        most_similarity = value[0]
        number = int(re.search(r'frame-0*(\d+)', most_similarity).group(1))
        next_= most_similarity.replace(str(number), str(number+1))
        last_= most_similarity.replace(str(number), str(number-1))
        if next_ not in pairs_path_dict:
            continue
        if last_ not in pairs_path_dict:
            continue
        new_value = [last_, next_]

        pairs_label_result[key] = new_value
        
    return pairs_path_dict, pairs_label_result


def process_scene(data_dir, pair_path, n):
    (datadir_dir, datadir_name, datadir_ext) = fio.get_filename_components(data_dir)
    print(cyan("Processing conversion for scene: {}".format(datadir_name)))
    path_dict, relation_dict = parse_pairs_file(pair_path, data_dir, n)
    return path_dict, relation_dict


def get_label(image_path):
    # Split the path by '/' and take the last two elements
    parts = image_path.split('/')
    return '/'.join(parts[-2:])


def rank_and_pair(numbers):
    # Sort the list in ascending order
    sorted_numbers = sorted(numbers)
    # Generate pairs
    pairs = [[sorted_numbers[i], sorted_numbers[i+1]] for i in range(len(sorted_numbers)-1)]
    return pairs

def load_yaml(file_path):
    yaml = YAML()
    yaml.preserve_quotes = True
    with open(file_path, 'r') as file:
        try:
            data = yaml.load(file)
            return data
        except Exception as e:
            print(f"Error parsing YAML file: {e}")
            return None

def save_yaml(data, output_file_path):
    yaml = YAML()
    yaml.indent(mapping=2, sequence=4, offset=2)
    yaml.preserve_quotes = True
    yaml.width = 4096  # Prevent line wrapping
    
    with open(output_file_path, 'w') as file:
        try:
            yaml.dump(data, file)
            print(f"YAML file saved successfully to {output_file_path}")
        except Exception as e:
            print(f"Error saving YAML file: {e}")

def modify_specific_fields(data, datatag, scenetag, savedirr):
    tag = fio.sep.join([datatag, scenetag])
    savedirr = savedirr.replace(fio.getParentDir() + fio.sep, '')
    savedirr = savedirr.replace(fio.sep + 'test', '')

    # Modify wandb/tags
    if 'wandb' in data:
        if 'tags' in data['wandb']:
            data['wandb']['tags'] = [tag, '256x256']
        
        # Modify wandb/name
        if 'name' in data['wandb']:
            data['wandb']['name'] = tag

    # Modify dataset/roots
    if 'dataset' in data and 'roots' in data['dataset']:
        data['dataset']['roots'] = [savedirr]

    # Modify test/eval_time_skip_steps
    if 'test' in data and 'eval_time_skip_steps' in data['test']:
        data['test']['eval_time_skip_steps'] = 1

    return data


def generate_command(experiment, checkpoint, mode, index_path, num_context_views, compute_scores):
    command = f"""remove outputs/test \\
    python -m src.main +experiment={experiment} \\
    checkpointing.load={checkpoint} \\
    mode={mode} \\
    dataset/view_sampler=evaluation \\
    dataset.view_sampler.index_path={index_path}"""
    return command


def save_command_to_file(command, filename):
    with open(filename, 'w') as f:
        f.write("#!/bin/bash\n\n")  # Add shebang line
        f.write(command)
    fio.os.chmod(filename, 0o755)  # Make the file executable


if __name__ == '__main__':
    data_tag = '7s'
    # scene_tag = 'scene_stairs'
    scene_tag = 'scene_fire'

    # data_tag = 'camb'
    # scene_tag = 'scene_KingsCollege'
    
    this_time = fio.get_current_timestamp("%Y_%m_%d")
    sample_num_required = 2

    scene_data_dir = fio.createPath(fio.sep, [fio.getParentDir(), 'datasets_raw', data_tag, scene_tag])
    scene_pair_path = fio.createPath(fio.sep, [fio.getParentDir(), 'datasets_pairs', data_tag, scene_tag], 'pairs-query-netvlad10.txt')
    save_dir = fio.createPath(fio.sep, [fio.getParentDir(), 'datasets', data_tag, 'n' + str(sample_num_required), scene_tag, 'test'])
    resize_save_dir = fio.createPath(fio.sep, [fio.getParentDir(), 'datasets_resize', data_tag, 'n' + str(sample_num_required), scene_tag])

    if fio.file_exist(save_dir):
        fio.delete_folder(save_dir)
    fio.ensure_dir(save_dir)

    if fio.file_exist(resize_save_dir):
        fio.delete_folder(resize_save_dir)
    fio.ensure_dir(resize_save_dir)

    yaml_input_fth = fio.createPath(fio.sep, [fio.getParentDir(), "config", "experiment"], "acid.yaml")
    yaml_output_fth = fio.createPath(fio.sep, [fio.getParentDir(), "config", "experiment"], "_".join([data_tag, scene_tag, 'n'+str(sample_num_required)]) + ".yaml")
    if fio.file_exist(yaml_output_fth):
        fio.delete_file(yaml_output_fth)
    yaml_content = load_yaml(yaml_input_fth)
    new_yaml_content = modify_specific_fields(yaml_content, data_tag, scene_tag, save_dir)
    save_yaml(yaml_content, yaml_output_fth)

    if (fio.file_exist(scene_data_dir) == False) or (fio.file_exist(scene_pair_path) == False):
        print(cyan("[ERROR] No data detected: {}, {}". format(data_tag, scene_tag)))
        exit()

    branch = 'train_full_byorder_85'
    train_intri_pth = fio.createPath(fio.sep,[scene_data_dir, branch, 'sparse', '0'], 'cameras.txt')
    train_intri_dict = get_colmap_intrinsic(train_intri_pth)
    train_extri_pth = fio.createPath(fio.sep,[scene_data_dir, branch, 'sparse', '0'], 'images.txt')
    train_extri_dict = get_colmap_extrinsic(train_extri_pth)
    train_pnt3d_pth = fio.createPath(fio.sep,[scene_data_dir, branch, 'sparse', '0'], 'points3D.txt')
    train_pnt3d_dict = get_colmap_points3d(train_pnt3d_pth)

    branch = 'test_full_byorder_59'
    test_intri_pth = fio.createPath(fio.sep,[scene_data_dir, branch, 'sparse', '0'], 'cameras.txt')
    test_intri_dict = get_colmap_intrinsic(test_intri_pth)
    test_extri_pth = fio.createPath(fio.sep,[scene_data_dir, branch, 'sparse', '0'], 'images.txt')
    test_extri_dict = get_colmap_extrinsic(test_extri_pth)

    chunk_size = 0
    chunk_index = 0
    chunk: list[Example] = []

    def save_chunk():
        global chunk_size
        global chunk_index
        global chunk

        chunk_key = f"{chunk_index:0>6}"
        print(
            # f"Saving chunk {chunk_key} of {len(keys)} ({chunk_size / 1e6:.2f} MB)."
                f"Saving chunk {chunk_key} ({chunk_size / 1e6:.2f} MB)."
        )
        dir = save_dir
        torch.save(chunk, dir + fio.sep + "{}.torch".format(chunk_key))
        chunk_size = 0
        chunk_index += 1
        chunk = []

    paths, relation = process_scene(scene_data_dir, scene_pair_path, n=sample_num_required)
    asset_dict = {}
    
    for key, base_labels in relation.items():
        
        vid_dict, intrinsics, world2cams, cam2worlds, near_fars = {}, {}, {}, {}, {}
        images_dict = {}
        base_vids = []
        num_bytes = 0

        vid = 0
        asset_target = vid
        key_img_path = paths[key]
        update_img_pth, vid_name, intrins, extrins, near, far = get_params(key_img_path, test_intri_dict, test_extri_dict)
        
        # images_dict[vid] = load_raw(key_img_path)
        resize_save_path = fio.createPath(fio.sep, [resize_save_dir], 'resize_' + key)
        (duedir, duename, dueext) = fio.get_filename_components(resize_save_path)
        fio.ensure_dir(duedir)
        resized_size_key, tensor_key = load_raw_resize(key_img_path, resize_save_path)
        images_dict[vid] = tensor_key

        vid_dict[vid] = vid_name
        intrinsics[vid] = intrins
        world2cams[vid] = extrins
        cam2worlds[vid] = np.linalg.inv(extrins)
        near_fars[vid] = [near, far]
        # num_bytes += get_size(key_img_path)
        num_bytes += resized_size_key
        
        for blabel in base_labels:
            vid += 1
            base_img_path = paths[blabel]
            update_img_pth, vid_name, intrins, extrins, near, far = get_params(base_img_path, train_intri_dict, train_extri_dict, train_pnt3d_dict)
        
            base_vids.append(vid)
            
            # images_dict[vid] = load_raw(base_img_path)
            resize_save_path = fio.createPath(fio.sep, [resize_save_dir], 'resize_' + blabel)
            (duedir, duename, dueext) = fio.get_filename_components(resize_save_path)
            fio.ensure_dir(duedir)
            resized_size_base, tensor_base = load_raw_resize(key_img_path, resize_save_path)
            images_dict[vid] = tensor_base

            vid_dict[vid] = vid_name

            intrinsics[vid] = intrins
            world2cams[vid] = extrins
            cam2worlds[vid] = np.linalg.inv(extrins)

            near_fars[vid] = [near, far]
            # num_bytes += get_size(base_img_path)
            num_bytes += resized_size_base

        asset_dict[key] = {
            "context": base_vids,
            "target": [asset_target]
        }

        example = load_metadata(intrinsics, world2cams)
        example["images"] = [
            images_dict[timestamp.item()] for timestamp in example["timestamps"]
        ]

        assert len(images_dict) == len(example["timestamps"])
        example["key"] = key
        
        print(f"    Added {key} to chunk ({num_bytes / 1e6:.2f} MB).")
        chunk.append(example)
        chunk_size += num_bytes

        if chunk_size >= TARGET_BYTES_PER_CHUNK:
            save_chunk()

    if chunk_size > 0:
        save_chunk()

    save_path_asset = fio.createPath(fio.sep, [save_dir], 'evaluation.json')
    with open(save_path_asset, 'w') as file:
        json.dump(asset_dict, file)

    index = {}
    save_path_index = fio.createPath(fio.sep, [save_dir], 'index.json')
    chunk_paths = fio.traverse_dir(save_dir, full_path=True, towards_sub=False)
    chunk_paths = fio.filter_ext(chunk_paths, filter_out_target=False, ext_set=['torch'])
    for chnk_pth in chunk_paths:
        (chnkfile, chnkname, chnkext) = fio.get_filename_components(chnk_pth)
        chunk = torch.load(chnk_pth)
        for example in chunk:
            index[example["key"]] = chnkname + '.' + chnkext

    with open(save_path_index, 'w') as f_index:
        json.dump(index, f_index)

    (yamldir, yamlname, yamlext) = fio.get_filename_components(yaml_output_fth)
    input_eval_index_path = fio.createPath(fio.sep, ['datasets', data_tag, 'n' + str(sample_num_required), scene_tag, 'test'], "evaluation.json")
    num_context_views = sample_num_required
    compute_score = True
    command = generate_command(yamlname, "checkpoints/acid.ckpt", "test", input_eval_index_path, num_context_views, compute_score)
    save_command_path = fio.createPath(fio.sep, [save_dir], "command.sh")
    save_command_to_file(command, save_command_path)