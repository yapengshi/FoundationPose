import threading
import time
import cv2
import numpy as np
import pyrealsense2 as rs
import numpy as np
import cv2
import json
import time
import os
import rospy
from std_msgs.msg import String  # 新增：用于发送信号
from estimater import *
from datareader import *
import argparse
# Initialize ROS node
rospy.init_node('realsense_camera')
signal_pub = rospy.Publisher('/camera/save_signal', String, queue_size=1)  # 新增：信号发布器
hztest_pub = rospy.Publisher('/foundationpose_hztest', String, queue_size=1)  # 新增：信号发布器
# Create a pipeline
pipeline = rs.pipeline()

# Create a config and configure the pipeline to stream
# different resolutions of color and depth streams
config = rs.config()

# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))

found_rgb = False
for s in device.sensors:
    if s.get_info(rs.camera_info.name) == "RGB Camera":
        found_rgb = True
        break
if not found_rgb:
    print("The demo requires Depth camera with Color sensor")
    exit(0)

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

if device_product_line == "L500":
    config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
else:
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
profile = pipeline.start(config)

# Getting the depth sensor's depth scale (see rs-align example for explanation)
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print("Depth Scale is: ", depth_scale)

# We will be removing the background of objects more than
#  clipping_distance_in_meters meters away
clipping_distance_in_meters = 1  # 1 meter
clipping_distance = clipping_distance_in_meters / depth_scale

# Create an align object
# rs.align allows us to perform alignment of depth frames to others frames
# The "align_to" is the stream type to which we plan to align depth frames.
align_to = rs.stream.color
align = rs.align(align_to)

# Get the absolute path to the subfolder
script_dir = os.path.dirname(os.path.abspath(__file__))
subfolder_depth = os.path.join(script_dir, "out1/depth")
subfolder_rgb = os.path.join(script_dir, "out1/rgb")
subfolder_depth_unaligned = os.path.join(script_dir, "out1/depth_unaligned")
subfolder_rgb_unaligned = os.path.join(script_dir, "out1/rgb_unaligned")
show_first_flag = False

# Check if the subfolder exists, and create it if it does not
if not os.path.exists(subfolder_depth):
    os.makedirs(subfolder_depth)
if not os.path.exists(subfolder_rgb):
    os.makedirs(subfolder_rgb)
if not os.path.exists(subfolder_depth_unaligned):
    os.makedirs(subfolder_depth_unaligned)
if not os.path.exists(subfolder_rgb_unaligned):
    os.makedirs(subfolder_rgb_unaligned)

RecordStream = False
first_frame_published = False  # Flag to track if first frame has been published
first_mask_recieved = False
foundation_start_flag = False
foundation_first_flag = False
click_x = None
click_y = None
wait_count = 20
mask_path = None
def mouse_callback(event, x, y, flags, param):
    global click_x, click_y, save_click
    if event == cv2.EVENT_LBUTTONDOWN:
        click_x, click_y = x, y
        save_click = True
        print(f"Mouse clicked at: ({x}, {y})")
        
def mask_callback(msg):
    global mask_path,first_mask_recieved
    mask_path = str(msg.data)
    print(f"receive:{mask_path}")
    first_mask_recieved = True
mask_sub = rospy.Subscriber("/sam/mask_path", String, mask_callback, queue_size=1)

def realsense_to_color(color_frame, target_size=(640, 480)):
    """将RealSense color_frame转换为与get_color()相同的格式"""
    color_image = np.asanyarray(color_frame.get_data())
    # 调整尺寸（如果需要）
    if (color_image.shape[1], color_image.shape[0]) != target_size:
        color_image = cv2.resize(color_image, target_size, interpolation=cv2.INTER_NEAREST)
    return color_image[..., :3]  # 确保只有RGB三个通道


def realsense_to_color(color_frame, target_size=(640, 480)):
    """将RealSense color_frame转换为与get_color()相同的格式"""
    color_image = np.asanyarray(color_frame.get_data())
    # 调整尺寸（如果需要）
    if (color_image.shape[1], color_image.shape[0]) != target_size:
        color_image = cv2.resize(color_image, target_size, interpolation=cv2.INTER_NEAREST)
    # 将 BGR 转换为 RGB
    color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
    return color_image[..., :3]  # 确保只有RGB三个通道

def realsense_to_depth(depth_frame, zfar=np.inf, target_size=(640, 480)):
    """将RealSense depth_frame转换为与get_depth()相同的格式"""
    depth_image = np.asanyarray(depth_frame.get_data())
    # 转换为米并应用深度比例
    #depth_scale = depth_frame.get_profile().as_video_stream_profile().get_device().first_depth_sensor().get_depth_scale()
    depth_meters = depth_image.astype(np.float32) * depth_scale
    if (depth_meters.shape[1], depth_meters.shape[0]) != target_size:
        depth_meters = cv2.resize(depth_meters, target_size, interpolation=cv2.INTER_NEAREST)
    
    # 应用与get_depth()相同的过滤
    depth_meters[(depth_meters < 0.001) | (depth_meters >= zfar)] = 0
    return depth_meters
def publish_pose_simple(pose_matrix, major_axis_index):
    """
    将4×4位姿矩阵展平为16长度数组并发布为字符串
    :param pose_matrix: 4×4 numpy数组
    """
    # 展平矩阵并格式化为字符串
    pose_flat = pose_matrix.flatten()
    pose_str = ' '.join(['%.6f' % x for x in pose_flat])  # 保留6位小数
    pose_str += ' ' + str(major_axis_index)
    hztest_pub.publish(pose_str)

def determine_major_axis(pose, box):
    lengths = np.abs(box[1] - box[0])
    major_axis_index = np.argmax(lengths)
    return major_axis_index

class DisplayThread(threading.Thread):
    def __init__(self, window_name):
        threading.Thread.__init__(self)
        self.window_name = window_name
        self.current_image = None
        self.lock = threading.Lock()
        self.key_pressed = None
        self.key_event = threading.Event()
        self.running = True

    def update_image(self, new_image):
        with self.lock:
            self.current_image = new_image.copy() if new_image is not None else None

    def get_key_press(self):
        with self.lock:
            if self.key_pressed is None:
                return None
            key = self.key_pressed
            self.key_pressed = None
            self.key_event.clear()
            return key

    def stop(self):
        self.running = False

    def run(self):
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, mouse_callback)  # 设置鼠标回调
        while self.running:
            with self.lock:
                if self.current_image is not None:
                    cv2.imshow(self.window_name, self.current_image)
                    
            
            key = cv2.waitKey(1)
            if key != -1:
                with self.lock:
                    self.key_pressed = key
                    self.key_event.set()
            
            time.sleep(0.001)
        
        #cv2.destroyAllWindows()
        cv2.destroyWindow(self.window_name)
    

display_thread = DisplayThread("GET (X,Y)")
display_thread.start()


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    code_dir = os.path.dirname(os.path.realpath(__file__))
    parser.add_argument('--mesh_file', type=str, default=f'{code_dir}/demo_data/mustard0/mesh/textured_simple.obj')
    parser.add_argument('--test_scene_dir', type=str, default=f'{code_dir}/demo_data/mustard0')
    parser.add_argument('--est_refine_iter', type=int, default=5)
    parser.add_argument('--track_refine_iter', type=int, default=2)
    parser.add_argument('--debug', type=int, default=1)
    parser.add_argument('--debug_dir', type=str, default=f'{code_dir}/debug')
    args = parser.parse_args()
    set_logging_format()
    set_seed(0)
    mesh = trimesh.load(args.mesh_file)
    debug = args.debug
    debug_dir = args.debug_dir
    os.system(f'rm -rf {debug_dir}/* && mkdir -p {debug_dir}/track_vis {debug_dir}/ob_in_cam')
    to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
    bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2,3)
    #print(f"bbox:{bbox}")
    scorer = ScorePredictor()
    refiner = PoseRefinePredictor()
    glctx = dr.RasterizeCudaContext()
    est = FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals, mesh=mesh, scorer=scorer, refiner=refiner, debug_dir=debug_dir, debug=debug, glctx=glctx)
    logging.info("estimator initialization done")
    #reader = YcbineoatReader(video_dir=args.test_scene_dir, shorter_side=None, zfar=np.inf)
    rate = rospy.Rate(30)
    #print(f"bbox:{bbox}")
    try:
        while not rospy.is_shutdown():
            wait_count = max(wait_count - 1, 0)   #等待一些图片之后再开始
            # Get frameset of color and depth
            frames = pipeline.wait_for_frames()
            #print("Q")
            # frames.get_depth_frame() is a 640x360 depth image

            # Align the depth frame to color frame
            aligned_frames = align.process(frames)

            # Get aligned frames
            aligned_depth_frame = aligned_frames.get_depth_frame()  # aligned_depth_frame is a 640x480 depth image
            color_frame = aligned_frames.get_color_frame()

            unaligned_depth_frame = frames.get_depth_frame()
            unaligned_color_frame = frames.get_color_frame()

            # Get instrinsics from aligned_depth_frame
            intrinsics = aligned_depth_frame.profile.as_video_stream_profile().intrinsics

            # Validate that both frames are valid
            if not aligned_depth_frame or not color_frame:
                continue

            depth_image = np.asanyarray(aligned_depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            # Remove background - Set pixels further than clipping_distance to grey
            grey_color = 153
            depth_image_3d = np.dstack((depth_image, depth_image, depth_image))  # depth image is 1 channel, color is 3 channels
            bg_removed = np.where(
                (depth_image_3d > clipping_distance) | (depth_image_3d <= 0),
                grey_color,
                color_image,
            )

            unaligned_depth_image = np.asanyarray(unaligned_depth_frame.get_data())
            unaligned_rgb_image = np.asanyarray(unaligned_color_frame.get_data())

            # Render images:
            #   depth align to color on left
            #   depth on right
            depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET
            )

            # 准备要显示的图像
            images = np.hstack((color_image, depth_colormap))

            # 更新显示线程中的图像
            if foundation_start_flag == False:
                display_thread.update_image(images)
            # 检查按键（从显示线程获取）
            framename = int(round(time.time() * 1000))
            if wait_count == 0 and first_frame_published == False:
                # 保存颜色和深度图，并通知sam已经拍到第一张图片
                if click_x == None and click_y == None:
                    print("x,y are none,please click object to get x,y")
                else:
                    first_frame_published = True
                    image_path_depth = os.path.join(subfolder_depth, f"{framename}.png")
                    image_path_rgb = os.path.join(subfolder_rgb, f"{framename}.png")
                    cv2.imwrite(image_path_depth, depth_image)
                    cv2.imwrite(image_path_rgb, color_image)
                    signal_pub.publish(str(os.path.join(subfolder_rgb, f"{framename}.png")) + " " + str(click_x) + " " + str(click_y))

            if first_frame_published == True and first_mask_recieved == True:
                #cv2.destroyAllWindows()
                if debug == 0:
                    display_thread.stop()
                    display_thread.join()
                foundation_start_flag = True

            if foundation_start_flag == True:
                
                logging.info(f'i:{framename}')
                # color_frame = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
                color = realsense_to_color(color_frame, target_size=(640,480))  # 使用您的目标尺寸
                depth = realsense_to_depth(aligned_depth_frame, zfar=np.inf, target_size=(640,480))
                intrinsics = aligned_depth_frame.profile.as_video_stream_profile().intrinsics
                K = np.array([
                        [intrinsics.fx, 0, intrinsics.ppx],
                        [0, intrinsics.fy, intrinsics.ppy],
                        [0, 0, 1]
                    ])
                
                if foundation_first_flag == False:
                    foundation_first_flag = True
                    #print(color,depth)
                    mask = cv2.imread(mask_path, cv2.INTER_NEAREST)
                    if len(mask.shape)==3:
                        for c in range(3):
                            if mask[...,c].sum()>0:
                                mask = mask[...,c]
                                break
                    mask = cv2.resize(mask, (640,480), interpolation=cv2.INTER_NEAREST).astype(bool).astype(np.uint8)
                    pose = est.register(K=K, rgb=color, depth=depth, ob_mask=mask, iteration=args.est_refine_iter)
                    if pose is None:  # 判断是否为None对象
                        raise RuntimeError("位姿估计失败，返回None")
                    # input()
                    if debug>=3:
                        m = mesh.copy()
                        m.apply_transform(pose)
                        m.export(f'{debug_dir}/model_tf.obj')
                        xyz_map = depth2xyzmap(depth, K)
                        valid = depth>=0.001
                        pcd = toOpen3dCloud(xyz_map[valid], color[valid])
                        o3d.io.write_point_cloud(f'{debug_dir}/scene_complete.ply', pcd)
                elif foundation_first_flag == True:
                    print("yes")
                    pose = est.track_one(rgb=color, depth=depth, K=K, iteration=args.track_refine_iter)

                os.makedirs(f'{debug_dir}/ob_in_cam', exist_ok=True)
                np.savetxt(f'{debug_dir}/ob_in_cam/{framename}.txt', pose.reshape(4,4))
                major_axis_index = determine_major_axis(pose, bbox)
                #print(major_axis_index)
                publish_pose_simple(pose, major_axis_index)
                if debug>=1:
                    # center_pose = pose@np.linalg.inv(to_origin)
                    center_pose = pose
                    vis = draw_posed_3d_box(K, img=color, ob_in_cam=center_pose, bbox=bbox)
                    vis = draw_xyz_axis(color, ob_in_cam=center_pose, scale=0.1, K=K, thickness=3, transparency=0, is_input_rgb=True)
                    display_thread.update_image(vis[...,::-1])
                    
                if debug>=2:
                    os.makedirs(f'{debug_dir}/track_vis', exist_ok=True)
                    imageio.imwrite(f'{debug_dir}/track_vis/{framename}.png', vis)
            rate.sleep()
    except :
        print("www")
    finally:
        display_thread.stop()
        display_thread.join()
        cv2.destroyAllWindows()
        pipeline.stop()