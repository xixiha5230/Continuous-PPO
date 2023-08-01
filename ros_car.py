"""
ROS小车底层驱动
"""

import math
import os
import threading
import time

import matplotlib.pyplot as plt
import numpy as np
import rospy
from geometry_msgs.msg import PoseWithCovarianceStamped, Twist

# from limo_base.msg import LimoStatus
from nav_msgs.msg import Odometry
from PIL import Image as PILImage
from rospy.numpy_msg import numpy_msg
from sensor_msgs.msg import Image, LaserScan
import random

LASER_SCAN_SIZE = 400
LASER_SCAN_MAX = 3.0


def resize_camera(im_np):
    im_pil = PILImage.fromarray(im_np).resize((84, 84))

    return np.array(im_pil)


def quart_to_rpy(x, y, z, w):
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = math.asin(2 * (w * y - x * z))
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (z * z + y * y))
    return roll, pitch, yaw


def ros_xy_to_plt(x, y):
    return y, x


class RosCar:
    _closed = False

    camera_height = None  # 图像高度 (ROS格式)
    camera_width = None  # 图像宽度 (ROS格式)
    camera_image = None  # 缓存的图像 (ROS格式)

    laser_scan = None  # 缓存的雷达数据 (ROS格式)
    laser_scan_intensity = None  # 缓存的雷达强度数据 (ROS格式)

    position = init_position = None  # 缓存的位置坐标、初始位置坐标 (ROS格式)
    yaw = init_yaw = None  # 缓存的角度、初始角度 (ROS格式)
    velocity = None  # 缓存的速度向量 (ROS格式)

    # control_mode = None  # 小车控制模式 (ROS格式)
    # motion_mode = None  # 小车运动模式 (ROS格式)/LM_691/cmd_vel
    ROS_NAMESPACE = "LM_691"

    def __init__(
        self, ROS_MASTER_URI=None, ROS_IP=None, motor_limit=0.25, repeat_action=False
    ):
        """
        Args:
            ROS_MASTER_URI: ROS MASTER 地址, 若为None, 则自动读取环境变量
            ROS_IP: ROS IP 地址, 若为None, 则自动读取环境变量
            repeat_action: 在没有动作指令的情况下, 是否一直重复上一时刻的动作
        """
        if ROS_MASTER_URI is not None:
            os.environ["ROS_MASTER_URI"] = ROS_MASTER_URI
        if ROS_IP is not None:
            os.environ["ROS_IP"] = ROS_IP
        self.repeat_action = repeat_action
        self.motor_limit = motor_limit

        self.cmd_pub = rospy.Publisher(
            f"/{self.ROS_NAMESPACE}/cmd_vel", Twist, queue_size=10
        )

        rospy.init_node("ros_server_listener", anonymous=True, disable_signals=True)

        rospy.Subscriber(
            f"/{self.ROS_NAMESPACE}/camera/image_raw",
            numpy_msg(Image),
            self._camera_callback,
        )
        rospy.Subscriber(
            f"/{self.ROS_NAMESPACE}/scan", numpy_msg(LaserScan), self._scan_callback
        )
        # rospy.Subscriber('/robot_pose_ekf/odom_combined', numpy_msg(PoseWithCovarianceStamped), self._odom_combined_callback)
        rospy.Subscriber(
            f"/{self.ROS_NAMESPACE}/odom", numpy_msg(Odometry), self._odom_callback
        )
        # rospy.Subscriber('limo_status', LimoStatus, self._limo_status_callback)

        self._ros2rl_laser_index = []  # ROS -> RL 雷达转换数组
        for i in range(LASER_SCAN_SIZE // 2, 0, -1):
            self._ros2rl_laser_index.append(i - 1)
            self._ros2rl_laser_index.append(LASER_SCAN_SIZE - i)

        self._ros2rl_convert_mat = np.array([[[0, 1], [-1, 0]]])  # ROS -> RL 坐标转换矩阵

        # 如果重复指令，则开启无限发送动作指令的线程
        if self.repeat_action:
            self.cmd = None
            t = threading.Thread(target=self._repeat_send_cmd_vel)
            t.start()

    def initialized(self):
        """
        小车是否已经初始化完毕（传感信息是否已经能够接收），建议循环等待
        """
        print("===初始化检查开始===")
        print("camera_image:", self.camera_image is not None)
        print("laser_scan:", self.laser_scan is not None)
        print("position:", self.position is not None)
        print("yaw:", self.yaw is not None)
        print("velocity:", self.velocity is not None)
        print("===初始化检查结束===")

        return all(
            [
                self.camera_image is not None,
                self.laser_scan is not None,
                self.position is not None,
                self.yaw is not None,
                self.velocity is not None,
            ]
        )

    def _camera_callback(self, data: Image):
        """
        摄像机传感数据回调函数
        """
        im_np = np.frombuffer(data.data, dtype=np.uint8).reshape(
            data.height, data.width, -1
        )

        self.camera_height = data.height
        self.camera_width = data.width
        self.camera_image = im_np

    def get_camera_image(self):
        """
        获取摄像机传感数据 (ROS格式)
        大小为 (self.camera_height, self.camera_width, 3)
        """
        return self.camera_image

    def _scan_callback(self, data: LaserScan):
        """
        雷达传感数据回调函数
        """
        self.laser_scan_min_rad = data.angle_min
        self.laser_scan_max_rad = data.angle_max
        self.laser_scan = data.ranges
        self.laser_scan_intensity = data.intensities

    def get_laser_scan(self):
        """
        获取雷达传感数据 (ROS格式)
        一维向量, 长度不定, 在400左右
        """
        return self.laser_scan

    def _odom_callback(self, data: Odometry):
        """
        小车自身信息回调函数
        """
        position = data.pose.pose.position
        self.position = np.array([position.x, position.y])

        orientation = data.pose.pose.orientation
        roll, pitch, yaw = quart_to_rpy(
            orientation.x, orientation.y, orientation.z, orientation.w
        )
        self.yaw = yaw

        if self.init_position is None or self.init_yaw is None:
            self.reset()

        linear = data.twist.twist.linear
        self.velocity = np.array([linear.x, linear.y])

    def get_pose(self):
        """
        获取小车自身信息 (ROS格式)
        [速度x, 速度y, 角度, 位置x, 位置y]
        """
        return np.concatenate([self.velocity, np.array([self.yaw]), self.position])

    # def _limo_status_callback(self, data: LimoStatus):
    #     """
    #     limo小车独有状态信息回调函数, 获取小车运动模式
    #     """
    #     if data.control_mode != self.control_mode:
    #         self.control_mode = data.control_mode
    #         if data.control_mode == 1:
    #             print('ROS控制')
    #         elif data.control_mode == 2:
    #             print('手柄控制')
    #         else:
    #             print('CONTROL MODE UNKNOWN', data.motion_mode)
    #     if data.motion_mode != self.motion_mode:
    #         self.motion_mode = data.motion_mode
    #         if data.motion_mode == 0:
    #             print('四轮差速')
    #         elif data.motion_mode == 1:
    #             print('阿克曼')
    #         elif data.motion_mode == 2:
    #             print('麦轮')
    #         else:
    #             print('MODE_UNKNOWN')

    def _repeat_send_cmd_vel(self):
        """
        发送运动指令线程, 每秒10帧数据
        """
        r = rospy.Rate(10)
        while not self._closed:
            if self.cmd is not None:
                self.cmd_pub.publish(self.cmd)

            r.sleep()

    def _get_processed_laser_scan(self):
        """
        获取处理后的雷达数据, 统一到400条射线 (ROS格式)
        """
        laser_scan = self.laser_scan
        laser_scan_intensity = self.laser_scan_intensity
        laser_scan_size = len(laser_scan)
        size_delta = LASER_SCAN_SIZE - laser_scan_size
        if size_delta > 0:  # 如果真实的射线数较少
            idx = [0] * math.floor(size_delta / 2) + [laser_scan_size] * math.ceil(
                size_delta / 2
            )
            laser_scan = np.insert(laser_scan, idx, 0.03)
            laser_scan_intensity = np.insert(laser_scan_intensity, idx, 0.0)
        elif size_delta < 0:  # 如果真实的射线数较多
            idx = np.concatenate(
                [
                    np.arange(math.floor(-size_delta / 2)),
                    laser_scan_size - (np.arange(math.ceil(-size_delta / 2)) + 1),
                ]
            )
            laser_scan = np.delete(laser_scan, idx)
            laser_scan_intensity = np.delete(laser_scan_intensity, idx)

        return laser_scan, laser_scan_intensity

    def move_car(self, x, angle):
        """
        发送移动小车的动作指令 (ROS指令)
        """
        cmd = Twist()
        cmd.linear.x = x
        cmd.angular.z = angle
        # print(f'x: {x:2f}, az: {angle:2f}')

        if self.repeat_action:
            self.cmd = cmd
        else:
            self.cmd_pub.publish(cmd)
            # print("send action ros")

    def send_rl_action(self, action):
        """
        发送移动小车的强化学习动作指令 (RL指令)
        [方向, 油门]
        方向 in [-1, 1], -1表示向左, 1表示向右
        油门 in [-1, 1], -1表示向后, 1表示向前
        """
        action = action[0]
        steer, motor = action
        if motor >= 0:
            steer = -steer

        self.move_car(motor * self.motor_limit, steer)

    def get_rl_obs_shapes(self):
        """
        获取强化学习适用的观测维度 (RL格式)
        (摄像头, 雷达, 小车自身信息)
        """
        return [(84, 84, 3), ((LASER_SCAN_SIZE + 1) * 2,), (6,)]

    def get_rl_action_size(self):
        """
        获取强化学习适用的动作维度 (RL格式)
        """
        return 2

    def get_rl_obs_list(self):
        """
        获取强化学习适用的观测值 (RL格式)
        (摄像头数据, 雷达数据, 小车自身信息数据)
        雷达为unity格式, 没有探测到的射线一律设置(1,1)
        小车自身信息为 (速度z, 速度x, 方向向量z, 方向向量x, 位置z, 位置x)
            方向向量与位置均为相对初始方向向量与初始位置的相对量，可通过 `reset` 重置初始状态
        所有维度与 `get_rl_obs_shapes` 一致
        """
        # CAMERA
        ros_camera = self.camera_image
        rl_camera_obs = np.array(resize_camera(ros_camera) / 255.0).astype(np.float32)

        # LASER
        ros_laser_scan, ros_laser_scan_intensity = self._get_processed_laser_scan()
        # rl_laser_scan = ros_laser_scan[self._ros2rl_laser_index]
        # rl_laser_scan_intensity = ros_laser_scan_intensity[self._ros2rl_laser_index]
        # rl_laser_scan = [[0. if d < LASER_SCAN_MAX and i > 1000 else 1.,
        #                  min(d / LASER_SCAN_MAX, 1.) if i > 1000 else 1.]
        #                  for d, i in zip(rl_laser_scan, rl_laser_scan_intensity)]
        # rl_laser_scan = np.concatenate([[0., 0.], *rl_laser_scan]).astype(np.float32)
        rl_laser_scan = [
            1.0 if d > LASER_SCAN_MAX else d / LASER_SCAN_MAX for d in ros_laser_scan
        ]
        rl_laser_scan_obs = np.array(rl_laser_scan).astype(np.float32)

        # RELATIVE POSE
        # ros_velocity = self.velocity
        # ros_yaw = self.yaw - self.init_yaw
        # ros_position = self.position - self.init_position
        # rotate_matrix = np.array([[math.cos(self.init_yaw), -math.sin(self.init_yaw)],
        #                           [math.sin(self.init_yaw), math.cos(self.init_yaw)]])
        # ros_position = np.dot(ros_position, rotate_matrix)
        # rl_velocity = np.matmul(
        #     ros_velocity, self._ros2rl_convert_mat).squeeze(0)
        # rl_yaw = -ros_yaw
        # rl_forward = np.array([math.sin(rl_yaw), math.cos(rl_yaw)])
        # rl_position = np.matmul(
        #     ros_position, self._ros2rl_convert_mat).squeeze(0)
        # rl_pose = np.concatenate(
        #     [rl_velocity, rl_forward, rl_position]).astype(np.float32)
        # rl_pose_obs = np.expand_dims(rl_pose, 0)

        return [rl_camera_obs, rl_laser_scan_obs]

    def init_ros_plt(self):
        """
        初始化ROS原始传感数据的可视化界面
        """
        self._ros_fig, axes = plt.subplots(ncols=2)
        ax1, ax2 = axes
        ax1.axis("off")
        self._ros_im = ax1.imshow(np.zeros((self.camera_height, self.camera_width, 3)))

        ax2.spines["right"].set_visible(False)
        ax2.spines["top"].set_visible(False)
        ax2.spines["left"].set_position("center")
        ax2.spines["bottom"].set_position("center")
        ax2.set_xlim(-LASER_SCAN_MAX, LASER_SCAN_MAX)
        ax2.set_ylim(-LASER_SCAN_MAX, LASER_SCAN_MAX)
        ax2.invert_xaxis()
        (self._ros_pos_hl,) = ax2.plot([], [], linewidth=3, color="r")
        self._ros_pos_hl_sc = ax2.scatter([], [], s=50, color="r", marker="o")
        (self._ros_vel_hl,) = ax2.plot([], [], color="y")
        self._ros_laser_sc = ax2.scatter([], [], s=5, color="b")

        self._ros_fig.canvas.draw()

    def update_ros_plt(self):
        """
        更新ROS原始传感数据的可视化界面
        """
        # CAMERA
        self._ros_im.set_data(self.camera_image)

        # RELATIVE POSE
        #  YAW
        yaw = self.yaw
        yaw = yaw - self.init_yaw
        #  POSITION
        position = self.position
        position = position - self.init_position
        rotate_matrix = np.array(
            [
                [math.cos(self.init_yaw), -math.sin(self.init_yaw)],
                [math.sin(self.init_yaw), math.cos(self.init_yaw)],
            ]
        )
        position = np.dot(position, rotate_matrix)

        d = 0.5
        pos_yaw = np.array([math.cos(yaw) * d, math.sin(yaw) * d]) + position
        pos_x, pos_y = position
        pos_yaw_x, pos_yaw_y = pos_yaw
        pos_x_converted, pos_y_converted = ros_xy_to_plt(pos_x, pos_y)
        pos_yaw_x_converted, pos_yaw_y_converted = ros_xy_to_plt(pos_yaw_x, pos_yaw_y)
        self._ros_pos_hl.set_xdata([pos_x_converted, pos_yaw_x_converted])
        self._ros_pos_hl.set_ydata([pos_y_converted, pos_yaw_y_converted])
        self._ros_pos_hl_sc.set_offsets([pos_x_converted, pos_y_converted])

        vel_x, vel_y = self.velocity
        vel_x_converted, vel_y_converted = ros_xy_to_plt(vel_x, vel_y)
        self._ros_vel_hl.set_xdata([0, vel_x_converted])
        self._ros_vel_hl.set_ydata([0, vel_y_converted])

        # LASER
        laser_scan, laser_scan_intensity = self._get_processed_laser_scan()
        laser_scan = [
            d if i > 1000 else 1000 for d, i in zip(laser_scan, laser_scan_intensity)
        ]
        laser_rad = np.linspace(
            self.laser_scan_min_rad + yaw,
            self.laser_scan_max_rad + yaw,
            LASER_SCAN_SIZE,
        )
        laser_x = np.cos(laser_rad) * laser_scan + pos_x
        laser_y = np.sin(laser_rad) * laser_scan + pos_y
        laser_x_converted, laser_y_converted = ros_xy_to_plt(laser_x, laser_y)
        self._ros_laser_sc.set_offsets(np.c_[laser_x_converted, laser_y_converted])

        self._ros_fig.canvas.flush_events()

    def stop(self):
        """
        强制停止小车
        """
        self.move_car(0, 0)

    def reset(self):
        """
        重置小车的初始状态
        """
        self.init_position = self.position
        self.init_yaw = self.yaw
        self.stop()

    def is_shutdown(self):
        return rospy.is_shutdown()

    def close(self, reason=""):
        """
        关闭ROS数据接收
        """
        self._closed = True
        rospy.signal_shutdown(reason)

    def env_step(self, action):
        self.send_rl_action(action)
        state = self.get_rl_obs_list()
        reward = []
        done = False
        _ = None
        return state, reward, done, _


if __name__ == "__main__":
    # plt.ion()
    car = RosCar()
    # test = True
    # while test:
    #     car.move_car(1,1)
    #     print("开始")
    while not car.initialized():
        time.sleep(0.5)
    print("car initialized")
    print("laser_scan_size", len(car.laser_scan))
    if LASER_SCAN_SIZE != len(car.laser_scan):
        print(f"warning! laser_scan_size != {LASER_SCAN_SIZE}")
    # print('laser_scan_min_rad', car.laser_scan_min_rad)
    # print('laser_scan_max_rad', car.laser_scan_max_rad)
    # car.init_ros_plt()
    while not car.is_shutdown():
        # car.update_ros_plt()
        state = car.get_rl_obs_list()
        speed = (1 - (-1)) * np.random.random() + (-1)
        angele = (1 - (-1)) * np.random.random() + (-1)
        print("speed:", speed, " angle:", angele)
        car.move_car(speed, angele)
