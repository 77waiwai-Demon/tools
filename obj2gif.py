import numpy as np
import trimesh
import pyrender
import os
import imageio
import argparse
from skimage.color import rgb2hsv, hsv2rgb
from PIL import Image

def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--obj_path", type=str)
    parser.add_argument("--gif_path", type=str)
    parser.add_argument("--frames_folder_path", type=str)
    parser.add_argument("--image_width", type=int, default=300)
    parser.add_argument("--image_height", type=int, default=300)
    parser.add_argument("--num_frames", type=int, default=36,
                        help="number of rendered images to create the GIF")
    parser.add_argument("--image_duration", type=float, default=0.05,
                        help="duration of each image in the GIF")
    parser.add_argument("--image_loops", type=float, default=0,
                        help="number of loops in the GIF (set as 0 to allow looping endlessly)")
    parser.add_argument("--set_initial_camera_pose", type=bool, default=False)
    parser.add_argument("--camera_pose", type=str,
                        default="[[1.0,0.0,0.0,0.0],[0.0,1.0,0.0,0.0],[0.0,0.0,1.0,0.0],[0.0,0.0,0.0,1.0]]",
                        help="camera pose (4*4 matrix) for the first image in the GIF")
    parser.add_argument("--light_intensity", type=float, default=3.0,
                        help="brightness of light, in lux (lm/m^2).")
    parser.add_argument("--light_color", type=str, default="[1.0,1.0,1.0]",
                        help="RGB value for the light’s color in linear space")
    parser.add_argument("--image_saturation", type=float, default=1.0,
                        help="saturation factor of the GIF. It's observed that sometimes rendered images are lower in saturation than expected")


    args = parser.parse_args()
    # preprocess camera_pose to correct format
    if args.set_initial_camera_pose:
        args.camera_pose=eval(args.camera_pose)
        args.camera_pose=np.array(args.camera_pose)
    args.light_color = eval(args.light_color)
    return args

def init_scene(args, rotate_x_axis=False):
    # load mesh to the scene
    mesh_trimesh = trimesh.load(args.obj_path)
    # x轴旋转-90度
    if rotate_x_axis:
        mesh_trimesh.apply_transform(trimesh.transformations.rotation_matrix(np.radians(90), [1, 0, 0]))
    mesh_pyrender = pyrender.Mesh.from_trimesh(mesh_trimesh)
    scene = pyrender.Scene()
    mesh_node = scene.add(mesh_pyrender)
    return scene,mesh_node

def change_parameters(args):
    # initialize scene
    scene, mesh_node=init_scene(args)

    # set up the initial camera pose (if any)
    if args.set_initial_camera_pose:
        camera = pyrender.PerspectiveCamera(yfov=np.pi / 2.7)
        scene.add(camera, pose=args.camera_pose)

    # set up the initial light
    light = pyrender.DirectionalLight(color=args.light_color, intensity=args.light_intensity)
    if args.set_initial_camera_pose:
        light_node = scene.add(light, pose=args.camera_pose)
    else:
        light_node = scene.add(light)

    # initialize the viewer
    viewer = pyrender.Viewer(scene, use_raymond_lighting=True, run_in_thread=True,viewport_size=(args.image_width,args.image_height))

    # choose the camera pose from the viewer
    input("Adjust the camera in the viewer, then press Enter to get the camera pose...")
    camera_nodes = [node for node in scene.get_nodes() if isinstance(node.camera, pyrender.PerspectiveCamera)]
    if args.set_initial_camera_pose:
        if np.abs(np.sum(np.abs(np.array(camera_nodes[0].matrix))-np.abs(args.camera_pose)))>0.000001:
            new_camera_pose = camera_nodes[0].matrix
        else:
            new_camera_pose = camera_nodes[1].matrix
    else:
        print(camera_nodes)
        new_camera_pose = camera_nodes[0].matrix
    formatted_new_camera_pose = np.array2string(new_camera_pose, separator=', ',
                                                formatter={'float_kind': lambda x: f"{x:.6f}"})
    print("Chosen camera pose Matrix:")
    print(formatted_new_camera_pose)

    # choose the light intensity from the viewer
    new_light_intensity=args.light_intensity
    count=0
    while True:
        intensity_input = input("Enter new light intensity number (>0) or 'q' to quit: ")
        if intensity_input.lower() == 'q':
            break
        try:
            count=count+1
            # Update light intensity
            new_light_intensity = float(intensity_input)
            light.intensity = new_light_intensity
            # Re-render the scene
            viewer.render_lock.acquire()
            scene.set_pose(light_node, pose=new_camera_pose)
            viewer.render_lock.release()
        except ValueError:
            print("Invalid input. Please enter a valid number or 'q' to quit.")
    if count==0:
        new_light_intensity=args.light_intensity
    print("Chosen light intensity:")
    print(new_light_intensity)

    # close the viewer
    viewer.close_external()
    del viewer
    return new_camera_pose,new_light_intensity

def change_saturation(image, factor):
    hsv_image = rgb2hsv(image)
    hsv_image[..., 1] *= factor
    hsv_image[..., 1] = np.clip(hsv_image[..., 1], 0, 1)
    rgb_image = hsv2rgb(hsv_image) * 255
    rgb_image = np.clip(rgb_image, 0, 255).astype(np.uint8)
    return rgb_image

def generate_gif(new_camera_pose, new_light_intensity, args):
    # initialize scene
    scene, mesh_node = init_scene(args)
    # Set up the camera = 
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.5)
    scene.add(camera, pose=new_camera_pose)
    # Set up the light
    light = pyrender.DirectionalLight(color=np.ones(3), intensity=new_light_intensity)
    scene.add(light, pose=new_camera_pose)

    # make directory to save frames
    os.makedirs(args.frames_folder_path, exist_ok=True)
    # render the scene from different angles
    r = pyrender.OffscreenRenderer(args.image_width, args.image_height)
    angle_step = 360 / args.num_frames  # Calculate the angle step for smooth rotation
    for i in range(args.num_frames):
        # rotate the model around the Y-axis
        angle = np.radians(i * angle_step) -180  # Make sure to rotate horizontally
        rotation_matrix = np.array([
            [np.cos(angle), 0, np.sin(angle), 0],
            [0, 1, 0, 0],  # Keep Y-axis fixed, rotate around Y-axis
            [-np.sin(angle), 0, np.cos(angle), 0],
            [0, 0, 0, 1]
        ])
        mesh_pose = np.dot(rotation_matrix, np.eye(4))
        scene.set_pose(mesh_node, pose=mesh_pose)
        # render
        color, depth = r.render(scene)
        alpha = np.clip(depth, 0, 1) * 255  # 将深度值映射到[0, 1]范围内，确保没有负值
        # 将alpha通道添加到color图像中，假设color的形状为 (H, W, 3)，alpha的形状为 (H, W)
        # import cv2
        # color = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
        color = change_saturation(color, args.image_saturation)
        color = np.dstack((color, alpha))  # 添加alpha通道到color
        color = np.clip(color, 0, 255).astype(np.uint8)
        # save the rendered frame
        frame_path = os.path.join(args.frames_folder_path, f'frame_{i:03d}.png')
        imageio.imwrite(frame_path, color)
        # cv2.imwrite(frame_path, color)
        if i == 0:
            # 存放在obj的目录下
            imageio.imwrite(os.path.join(args.obj_path[:-9], 'rmbg.png'), color)

    # create a GIF from the rendered frames
    frames = []
    pil_list = []
    for i in range(args.num_frames):
        frame_path = os.path.join(args.frames_folder_path, f'frame_{i:03d}.png')
        pil = Image.open(frame_path)
        pil_list.append(pil)
        frames.append(imageio.imread(frame_path))
    # imageio.mimsave(args.gif_path, frames, duration=args.image_duration, loop=args.image_loops)
    pil_list[0].save(args.gif_path, save_all=True, append_images=pil_list[1:],duration=1,transparency=0,loop=0,disposal=2)
    print(f"GIF saved to {args.gif_path}")



if __name__ == "__main__":
    args = init_args()
    dir_list = os.listdir('.')
    for dir in dir_list:
        if os.path.isdir(dir):
            try:
                args.obj_path = dir + '/model.obj'
                args.gif_path = dir + '/model.gif'
                args.frames_folder_path = dir + '/frames'
                # 删除当前目录下的gif文件
                sub_files = os.listdir(dir)
                for sub_file in sub_files:
                    if sub_file.endswith('.gif'):
                        os.remove(os.path.join(dir, sub_file))
                    if sub_file == 'frames':
                        sub_files2 = os.listdir(os.path.join(dir, sub_file))
                        for sub_file2 in sub_files2:
                            os.remove(os.path.join(dir, sub_file, sub_file2))
                        # 删除空文件夹
                        os.rmdir(os.path.join(dir, sub_file))
                        
                # new_camera_pose,new_light_intensity=change_parameters(args)
                new_light_intensity = 10.0
                new_camera_pose = [[-0.940841, 0.043940, 0.335987, 34.813687],
                                    [0.017732, 0.996582, -0.080679, -8.367098],
                                    [-0.338384, -0.069948, -0.938405, -97.242653],
                                    [0.000000, 0.000000, 0.000000, 1.000000]]
                generate_gif(new_camera_pose,new_light_intensity,args)
                print('finish:', dir)
                for sub_file in sub_files:
                    if sub_file == 'frames':
                        sub_files2 = os.listdir(os.path.join(dir, sub_file))
                        for sub_file2 in sub_files2:
                            os.remove(os.path.join(dir, sub_file, sub_file2))
                        os.rmdir(os.path.join(dir, sub_file))
                        
            except Exception as e:
                print('error:', dir, e)
                continue





