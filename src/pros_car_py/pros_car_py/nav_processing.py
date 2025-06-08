"""
導航決策邏輯的核心，決定自走車下一步該採取哪種動作（如直行、左轉、原地旋轉等）
為 car_controller 提供動作指令
"""

from pros_car_py.nav2_utils import (
    get_yaw_from_quaternion,
    get_direction_vector,
    get_angle_to_target,
    calculate_angle_point,
    cal_distance,
)
import math
import time
import numpy as np
from enum import Enum, auto
import random



SAFE_DIST = 130.0 # 大於 130.0 代表可前進
OBSTACLE_DIST = 80.0 # 小於 80.0 代表有障礙
ROTATION_MIN_DURATION = 0.5
ROTATION_MAX_DURATION = 1.5
ROTATION_FIXED_DURATION = 1.0
FORWARD_DURATION = 3.0
BACKWARD_DURATION = 0.5
STOP_DURATION = 0.5
ROTATE_MAX = 5
D_YOLO_ALIGNED = 3.0# 單位為像素
STUCK_LIDAR_THERES = 10
STUCK_LIDAR_DURATION = 2.0

class FSMState(Enum):
    IDLE = auto()
    ROTATE = auto()
    CHECK_SAFE = auto()
    FORWARD = auto()
    STOP = auto()
    BACKWARD = auto()
    ALIGN_TARGET = auto()
    APPROACH_TARGET = auto()
    FINISH = auto()

class Nav2Processing:
    def __init__(self, ros_communicator, data_processor):
        self.ros_communicator = ros_communicator
        self.data_processor = data_processor
        self.finishFlag = False
        self.global_plan_msg = None
        self.index = 0
        self.index_length = 0
        self.recordFlag = 0
        self.goal_published_flag = False

        # FSM
        self.state = FSMState.IDLE
        self.last_detected_offset = 0.0
        self.rotate_direction = 1 # 1: counterclockwise, -1: clockwise
        self.last_rotate_direction = 1

        self.state_duration = 0.0
        self.fsm_start_time = time.time()
        self.rotation_counter = 0
        self.stop_counter = 0
        self.backward_flag = 0
        self.last_lidar_range = None
        self.stuck_start_time = None
        self.stuck_flag = 0
        

    def reset_nav_process(self):
        self.finishFlag = False
        self.recordFlag = 0
        self.goal_published_flag = False

    def finish_nav_process(self):
        self.finishFlag = True
        self.recordFlag = 1

    def get_finish_flag(self):
        return self.finishFlag

    def get_action_from_nav2_plan(self, goal_coordinates=None):
        if goal_coordinates is not None and not self.goal_published_flag:
            self.ros_communicator.publish_goal_pose(goal_coordinates)
            self.goal_published_flag = True
        orientation_points, coordinates = (
            self.data_processor.get_processed_received_global_plan()
        )
        action_key = "STOP"
        if not orientation_points or not coordinates:
            action_key = "STOP"
        else:
            try:
                z, w = orientation_points[0]
                plan_yaw = get_yaw_from_quaternion(z, w)
                car_position, car_orientation = (
                    self.data_processor.get_processed_amcl_pose()
                )
                car_orientation_z, car_orientation_w = (
                    car_orientation[2],
                    car_orientation[3],
                )
                goal_position = self.ros_communicator.get_latest_goal()
                target_distance = cal_distance(car_position, goal_position)
                if target_distance < 0.5:
                    action_key = "STOP"
                    self.finishFlag = True
                else:
                    car_yaw = get_yaw_from_quaternion(
                        car_orientation_z, car_orientation_w
                    )
                    diff_angle = (plan_yaw - car_yaw) % 360.0
                    if diff_angle < 30.0 or (diff_angle > 330 and diff_angle < 360):
                        action_key = "FORWARD"
                    elif diff_angle > 30.0 and diff_angle < 180.0:
                        action_key = "COUNTERCLOCKWISE_ROTATION"
                    elif diff_angle > 180.0 and diff_angle < 330.0:
                        action_key = "CLOCKWISE_ROTATION"
                    else:
                        action_key = "STOP"
            except:
                action_key = "STOP"
        return action_key

    def get_action_from_nav2_plan_no_dynamic_p_2_p(self, goal_coordinates=None):
        if goal_coordinates is not None and not self.goal_published_flag:
            self.ros_communicator.publish_goal_pose(goal_coordinates)
            self.goal_published_flag = True

        # 只抓第一次路径
        if self.recordFlag == 0:
            if not self.check_data_availability():
                return "STOP"
            else:
                print("Get first path")
                self.index = 0
                self.global_plan_msg = (
                    self.data_processor.get_processed_received_global_plan_no_dynamic()
                )
                self.recordFlag = 1
                action_key = "STOP"

        car_position, car_orientation = self.data_processor.get_processed_amcl_pose()

        goal_position = self.ros_communicator.get_latest_goal()
        target_distance = cal_distance(car_position, goal_position)

        # 抓最近的物標(可調距離)
        target_x, target_y = self.get_next_target_point(car_position)

        if target_x is None or target_distance < 0.5:
            self.ros_communicator.reset_nav2()
            self.finish_nav_process()
            return "STOP"

        # 計算角度誤差
        diff_angle = self.calculate_diff_angle(
            car_position, car_orientation, target_x, target_y
        )
        if diff_angle < 20 and diff_angle > -20:
            action_key = "FORWARD"
        elif diff_angle < -20 and diff_angle > -180:
            action_key = "CLOCKWISE_ROTATION"
        elif diff_angle > 20 and diff_angle < 180:
            action_key = "COUNTERCLOCKWISE_ROTATION"
        return action_key

    def check_data_availability(self):
        return (
            self.data_processor.get_processed_received_global_plan_no_dynamic()
            and self.data_processor.get_processed_amcl_pose()
            and self.ros_communicator.get_latest_goal()
        )

    def get_next_target_point(self, car_position, min_required_distance=0.5):
        """
        選擇距離車輛 min_required_distance 以上最短路徑然後返回 target_x, target_y
        """
        if self.global_plan_msg is None or self.global_plan_msg.poses is None:
            print("Error: global_plan_msg is None or poses is missing!")
            return None, None
        while self.index < len(self.global_plan_msg.poses) - 1:
            target_x = self.global_plan_msg.poses[self.index].pose.position.x
            target_y = self.global_plan_msg.poses[self.index].pose.position.y
            distance_to_target = cal_distance(car_position, (target_x, target_y))

            if distance_to_target < min_required_distance:
                self.index += 1
            else:
                self.ros_communicator.publish_selected_target_marker(
                    x=target_x, y=target_y
                )
                return target_x, target_y

        return None, None

    def calculate_diff_angle(self, car_position, car_orientation, target_x, target_y):
        target_pos = [target_x, target_y]
        diff_angle = calculate_angle_point(
            car_orientation[2], car_orientation[3], car_position[:2], target_pos
        )
        return diff_angle

    def filter_negative_one(self, depth_list):
        return [depth for depth in depth_list if depth != -1.0]


    def camera_nav(self):
        """
        YOLO 目標資訊 (yolo_target_info) 說明：

        - 索引 0 (index 0)：
            - 表示是否成功偵測到目標
            - 0：未偵測到目標
            - 1：成功偵測到目標

        - 索引 1 (index 1)：
            - 目標的深度距離 (與相機的距離，單位為公尺)，如果沒偵測到目標就回傳 0
            - 與目標過近時(大約 40 公分以內)會回傳 -1

        - 索引 2 (index 2)：
            - 目標相對於畫面正中心的像素偏移量
            - 若目標位於畫面中心右側，數值為正
            - 若目標位於畫面中心左側，數值為負
            - 若沒有目標則回傳 0

        畫面 n 個等分點深度 (camera_multi_depth) 說明 :

        - 儲存相機畫面中央高度上 n 個等距水平點的深度值。
        - 若距離過遠、過近（小於 40 公分）或是實體相機有時候深度會出一些問題，則該點的深度值將設定為 -1。
        """
        yolo_target_info = self.data_processor.get_yolo_target_info()
        camera_multi_depth = self.data_processor.get_camera_x_multi_depth()
        if camera_multi_depth == None or yolo_target_info == None:
            return "STOP"

        camera_forward_depth = self.filter_negative_one(camera_multi_depth[7:13])
        camera_left_depth = self.filter_negative_one(camera_multi_depth[0:7])
        camera_right_depth = self.filter_negative_one(camera_multi_depth[13:20])

        action = "STOP"
        limit_distance = 0.7

        if all(depth > limit_distance for depth in camera_forward_depth):
            if yolo_target_info[0] == 1:
                if yolo_target_info[2] > 200.0:
                    action = "CLOCKWISE_ROTATION_SLOW"
                elif yolo_target_info[2] < -200.0:
                    action = "COUNTERCLOCKWISE_ROTATION_SLOW"
                else:
                    if yolo_target_info[1] < 0.8:
                        action = "STOP"
                    else:
                        action = "FORWARD_SLOW"
            else:
                action = "FORWARD"
        elif any(depth < limit_distance for depth in camera_left_depth):
            action = "CLOCKWISE_ROTATION"
        elif any(depth < limit_distance for depth in camera_right_depth):
            action = "COUNTERCLOCKWISE_ROTATION"
        return action

    def check_vehicle_stuck_by_lidar(self, lidar_forward):
        """
        回傳 True 表示卡住需要脫困
        條件: 前方最近距離的變化 < STUCK_LIDAR_THERES 持續 STUCK_LIDAR_DURATION
        """
        if not lidar_forward:
            self.stuck_start_time = None
            return False

        front_min = min(lidar_forward)
        now = time.time()

        # 初始化
        if self.last_lidar_range is None:
            self.last_lidar_range = front_min
            self.stuck_start_time = now
            return False

        if abs(front_min - self.last_lidar_range) < STUCK_LIDAR_THERES:
            if (now - self.stuck_start_time) >= STUCK_LIDAR_DURATION:
                self.stuck_start_time = None
                self.last_lidar_range = None
                return True
            return False
        else:
            # 有明顯距離變化, 重新初始化
            self.last_lidar_range = front_min
            self.stuck_start_time = now
            return False

    def camera_nav_unity(self):
        """
        - 索引 0 (index 0)：
            - 表示是否成功偵測到目標
            - 0：未偵測到目標
            - 1：成功偵測到目標

        - 索引 1 (index 1)：
            - 目標的深度距離 (與相機的距離，單位為公尺)，如果沒偵測到目標就回傳 0
            - 與目標過近時(大約 40 公分以內)會回傳 -1

        - 索引 2 (index 2)：
            - 目標相對於畫面正中心的像素偏移量
            - 若目標位於畫面中心左側，數值為正
            - 若目標位於畫面中心右側，數值為負
            - 若沒有目標則回傳 0
        """
        yolo_target_info = self.data_processor.get_yolo_target_info()

        """
        - 儲存相機畫面中央高度上 n 個等距水平點的深度值。
        - 若距離過遠、過近（小於 40 公分）或是實體相機有時候深度會出一些問題，則該點的深度值將設定為 -1。
        """
        camera_multi_depth = self.data_processor.get_camera_x_multi_depth()
        camera_multi_depth = list(
            map(lambda x: x * 100.0, self.data_processor.get_camera_x_multi_depth())
        )
        
        """
        - 自走車正面 lidar 偵測 91 個數值
        - combined_lidar_data[0:35] 是左側的距離
        - combined_lidar_data[35:56] 是前方的距離
        - combined_lidar_data[56:91] 是右側的距離
        - 單位是公尺
        )
        """
        combined_lidar_data = list(
            map(lambda x: x * 100.0, self.data_processor.get_processed_lidar())
        )
        combined_lidar_data = [lrange for lrange in combined_lidar_data if lrange > 10]
        if camera_multi_depth == None and yolo_target_info == None and combined_lidar_data == None:
            return "STOP"

        detected_target = (yolo_target_info[0] == 1)
        yolo_target_object_depth = yolo_target_info[1] * 100.0
        yolo_target_offset = yolo_target_info[2] * 100.0

        # camera_forward_depth = self.filter_negative_one(camera_multi_depth[7:14])
        # camera_left_depth = self.filter_negative_one(camera_multi_depth[0:7])
        # camera_right_depth = self.filter_negative_one(camera_multi_depth[14:20])
        # left_depth_min = np.min(camera_left_depth)
        # right_depth_min = np.min(camera_right_depth)

        lidar_left = self.filter_negative_one(combined_lidar_data[0:35])
        lidar_forward = self.filter_negative_one(combined_lidar_data[35:55])
        lidar_right = self.filter_negative_one(combined_lidar_data[55:90])
        left_narrow = any(lrange < OBSTACLE_DIST for lrange in lidar_left)
        right_narrow = any(lrange < OBSTACLE_DIST for lrange in lidar_right)
        left_max = np.max(lidar_left)
        front_mean = np.mean(lidar_forward)
        right_max = np.max(lidar_right)
        left_wide = any(lrange > SAFE_DIST for lrange in lidar_left) 
        right_wide = any(lrange > SAFE_DIST for lrange in lidar_right)
        prefer_left = left_max > right_max
        # action = "STOP"

        front_is_wide = all(lrange > SAFE_DIST for lrange in lidar_forward)
        front_is_enough_space = any(lrange > SAFE_DIST for lrange in lidar_forward)
        too_close = any(lrange < OBSTACLE_DIST for lrange in lidar_forward)

        limit_distance = 10.0
        target_distance = 5.0

        
        # =============================================================
        # FSM state: IDLE, ROTATE, CHECK_SAFE, FORWARD, STOP, BACKWARD, ALIGN_TARGET, APPROACH_TARGET
        # 狀態切換順序要按重要性排
        # IDLE -> ROTATE -> CHECK_SAFE
        # CHECK_SAVE(save) -> FORWARD -> ROTATE -> CHECK_SAFE -> ...
        # CHECK_SAVE(unsave) -> ROTATE -> CHECK_SAFE -> ...
        # FORWARD -> (if close to obstacle) STOP -> ROTATE -> ...
        # FORWARD ->  STOP -> (if stop for many time) BACKWARD -> ROTATE -> ...
        # ROTATE -> (detected) ALIGN_TARGET -> (aligned) APPROACH_TARGET -> (if close to obstacle) STOP -> ROTATE -> ...
        # =============================================================

        print(f"============== self.state {self.state} =============")
        now = time.time()
        dt = now - self.fsm_start_time
        
        match self.state:
            case FSMState.IDLE:
                self.state = FSMState.ROTATE
                self.state_duration = random.uniform(ROTATION_MIN_DURATION, ROTATION_MAX_DURATION)
                self.rotate_direction = random.choice([-1, 1])
                self.fsm_start_time = now
                print(f"RANDOM rotate_direction: {self.rotate_direction}")
                print(f"Next status: {self.state}")
                return "STOP"

            # 旋轉探索環境或避開障礙物
            case FSMState.ROTATE:
                if detected_target:
                    self.rotation_counter = 0
                    print(f"yolo_target_offset: {yolo_target_offset}")
                    self.last_detected_offset = yolo_target_offset
                    self.state = FSMState.ALIGN_TARGET
                    print(f"Next status: {self.state}")
                    return "STOP"

                if dt < self.state_duration:
                    # 有空間且前一個動作不是後退
                    if front_is_enough_space and self.backward_flag == 0:
                        self.rotation_counter = 0
                        print(f"lrange: {lidar_forward}")
                        self.state = FSMState.FORWARD
                        self.state_duration = FORWARD_DURATION
                        self.fsm_start_time = now
                        print(f"Next status: {self.state}")
                        return "FORWARD"
                    else:
                        print(f"Next status: {self.state}")
                        return "COUNTERCLOCKWISE_ROTATION" if self.rotate_direction == 1 else "CLOCKWISE_ROTATION"

                elif self.rotation_counter < ROTATE_MAX:
                    print(f"rotation_counter: {self.rotation_counter}")
                    self.rotation_counter += 1
                    self.state = FSMState.CHECK_SAFE
                    print(f"Next status: {self.state}")
                    return "STOP"
                else:
                    self.rotation_counter = 0
                    self.state = FSMState.STOP
                    print(f"Next status: {self.state}")
                    return "STOP"

            # 移動車輛去對齊 target 
            case FSMState.ALIGN_TARGET:
                if abs(yolo_target_offset) > D_YOLO_ALIGNED:
                    print(f"yolo_target_offset: {yolo_target_offset}")
                    self.rotate_direction = 1 if yolo_target_offset > 0 else -1
                    self.last_detected_offset = yolo_target_offset
                    if self.check_vehicle_stuck_by_lidar(lidar_forward):
                        self.state = FSMState.BACKWARD
                        self.state_duration = BACKWARD_DURATION
                        self.fsm_start_time = now
                        print(f"Next status: {self.state}")
                        return "BACKWARD"
                    print(f"Next status: {self.state}")
                    return "COUNTERCLOCKWISE_ROTATION_SLOW" if self.rotate_direction == 1 else "CLOCKWISE_ROTATION_SLOW"

                else:
                    print(f"lrange: {lidar_forward}")
                    self.state = FSMState.APPROACH_TARGET
                    print(f"Next status: {self.state}")
                    return "FORWARD"


            # 對齊後向 target 前進
            case FSMState.APPROACH_TARGET:
                if not detected_target:
                    if front_is_enough_space and front_mean > OBSTACLE_DIST:
                        print(f"lrange: {lidar_forward}")
                        if self.check_vehicle_stuck_by_lidar(lidar_forward):
                            self.state = FSMState.BACKWARD
                            self.state_duration = BACKWARD_DURATION
                            self.fsm_start_time = now
                            print(f"Next status: {self.state}")
                            return "BACKWARD"
                        print(f"Next status: {self.state}")
                        return "FORWARD_SLOW"
                    else:
                        print(f"self.last_detected_offset: {self.last_detected_offset}")
                        self.rotate_direction = 1 if self.last_detected_offset > 0 else -1
                        print(f"Next status: {self.state}")
                        return "COUNTERCLOCKWISE_ROTATION_SLOW" if self.rotate_direction == 1 else "CLOCKWISE_ROTATION_SLOW"
                if too_close and front_mean < OBSTACLE_DIST:
                    print(f"lrange: {lidar_forward}")
                    self.rotate_direction = 1 if prefer_left else -1
                    print(f"rotate_direction: {self.rotate_direction}, STUCK!!!")
                    print(f"Next status: {self.state}")
                    return "COUNTERCLOCKWISE_ROTATION_SLOW" if self.rotate_direction == 1 else "CLOCKWISE_ROTATION_SLOW"
                
                if abs(yolo_target_offset) > D_YOLO_ALIGNED:
                    print(f"yolo_target_offset: {yolo_target_offset}")
                    self.rotate_direction = 1 if yolo_target_offset > 0 else -1
                    self.last_detected_offset = yolo_target_offset
                    self.state = FSMState.ALIGN_TARGET
                    print(f"Next status: {self.state}")
                    return "COUNTERCLOCKWISE_ROTATION_SLOW" if self.rotate_direction == 1 else "CLOCKWISE_ROTATION_SLOW"

                if (yolo_target_object_depth > 10.0 and 
                    front_is_wide):
                    print(f"lrange: {lidar_forward}")
                    print(f"Next status: {self.state}")
                    return "FORWARD_SLOW"
                if detected_target and 0.0 < yolo_target_object_depth < 10.0:
                    print(f"yolo_target_object_depth:{yolo_target_object_depth}")
                    self.state = FSMState.FINISH
                    print(f"Next status: {self.state}")
                    return "STOP"
                else:
                    if self.check_vehicle_stuck_by_lidar(lidar_forward):
                        self.state = FSMState.BACKWARD
                        self.state_duration = BACKWARD_DURATION
                        self.fsm_start_time = now
                        print(f"Next status: {self.state}")
                        return "BACKWARD"
                    print(f"Next status: {self.state}")
                    return "FORWARD"

           # 檢查前方是否可通行
            case FSMState.CHECK_SAFE:
                if detected_target:
                    self.state = FSMState.ALIGN_TARGET
                    print(f"Next status: {self.state}")
                    return "STOP"
   
                if front_is_enough_space:
                    print(f"lidar_forward: {lidar_forward}")
                    self.state = FSMState.FORWARD
                    self.state_duration = FORWARD_DURATION
                    self.fsm_start_time = now
                    print(f"Next status: {self.state}")
                    return "FORWARD"
                else:
                    print(f"Not save")
                    self.state = FSMState.ROTATE
                    self.state_duration = ROTATION_FIXED_DURATION
                    self.rotate_direction = 1 if prefer_left else -1
                    self.fsm_start_time = now
                    print(f"RANDOM rotate_direction: {self.rotate_direction}")
                    print(f"Next status: {self.state}")
                    return "STOP"
            
            # 前進    
            case FSMState.FORWARD:
                if detected_target:
                    self.state = FSMState.ALIGN_TARGET
                    return "STOP"
                if too_close:
                    print(f"lidar_forward: {lidar_forward}")
                    self.state = FSMState.STOP
                    self.state_duration = STOP_DURATION
                    self.fsm_start_time = now
                    print(f"Next status: {self.state}")
                    return "STOP"
                if front_is_wide:
                    self.backward_flag = 0
                    print(f"lidar_forward: {lidar_forward}")
                    if dt < self.state_duration:
                        print(f"Next status: {self.state}")
                        return "FORWARD"
                if right_narrow or left_narrow:
                    print(f"lidar_right: {lidar_right}")
                    print(f"lidar_left: {lidar_left}")
                    self.state = FSMState.ROTATE
                    self.state_duration = random.uniform(ROTATION_MIN_DURATION, ROTATION_MAX_DURATION)
                    self.rotate_direction = 1 if prefer_left else -1
                    self.fsm_start_time = now
                    print(f"Next status: {self.state}")
                    return "STOP"
                if right_wide or left_wide:
                    print(f"lidar_right: {lidar_right}")
                    print(f"lidar_left: {lidar_left}")
                    self.state = FSMState.ROTATE
                    self.state_duration = random.uniform(ROTATION_MIN_DURATION, ROTATION_MAX_DURATION)
                    self.rotate_direction = 1 if prefer_left else -1
                    self.fsm_start_time = now
                    print(f"Next status: {self.state}")
                    return "STOP"
                else:
                    self.state = FSMState.ROTATE
                    self.state_duration = random.uniform(ROTATION_MIN_DURATION, ROTATION_MAX_DURATION)
                    self.rotate_direction = random.choice([-1, 1])
                    self.fsm_start_time = now
                    print(f"RANDOM rotate_direction: {self.rotate_direction}")
                    print(f"Next status: {self.state}")
                    return "STOP"

            # 緊急停止
            case FSMState.STOP:
                if dt < self.state_duration:
                    print(f"Next status: {self.state}")
                    return "STOP"
                else:
                    self.state = FSMState.BACKWARD
                    self.state_duration = BACKWARD_DURATION
                    self.fsm_start_time = now
                    print(f"Next status: {self.state}")
                    return "BACKWARD"
                
            # 後退
            case FSMState.BACKWARD:
                if dt < self.state_duration:
                    self.backward_flag = 1
                    print(f"Next status: {self.state}")
                    return "BACKWARD"
                elif right_narrow == True or left_narrow == True:
                    self.state = FSMState.ROTATE
                    self.state_duration = ROTATION_FIXED_DURATION
                    self.rotate_direction = 1 if right_narrow else -1
                    self.fsm_start_time = now
                    print(f"FIXED_DURATION: {self.state_duration}")
                    print(f"Next status: {self.state}")
                    return "STOP"
                else:
                    self.state = FSMState.ROTATE
                    self.state_duration = ROTATION_FIXED_DURATION
                    if self.last_detected_offset != 0:
                        self.rotate_direction = 1 if self.last_detected_offset > 0 else -1
                    else:
                        self.rotate_direction =  1 if prefer_left else -1
                    self.fsm_start_time = now
                    print(f"RANDOM rotate_direction: {self.rotate_direction}")
                    print(f"Next status: {self.state}")
                    return "STOP"

            case FSMState.FINISH:
                return "STOP"
        return "STOP"

    def stop_nav(self):
        return "STOP"
