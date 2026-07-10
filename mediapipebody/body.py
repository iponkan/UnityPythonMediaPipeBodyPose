# MediaPipe Body
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np

import cv2
import threading
import time
import global_vars 
import struct
import math
import socket


LANDMARK_SET_FREE = 0
LANDMARK_SET_ANCHORED = 1


def _encode_varint(value):
    value = int(value)
    encoded = bytearray()
    while value > 0x7F:
        encoded.append((value & 0x7F) | 0x80)
        value >>= 7
    encoded.append(value)
    return bytes(encoded)


def _encode_key(field_number, wire_type):
    return _encode_varint((field_number << 3) | wire_type)


def _encode_uint64(field_number, value):
    return _encode_key(field_number, 0) + _encode_varint(value)


def _encode_uint32(field_number, value):
    return _encode_key(field_number, 0) + _encode_varint(value)


def _encode_float(field_number, value):
    return _encode_key(field_number, 5) + struct.pack("<f", float(value))


def _encode_message(field_number, payload):
    return _encode_key(field_number, 2) + _encode_varint(len(payload)) + payload


def _encode_landmark(index, x, y, z):
    # message Landmark { uint32 index = 1; float x = 2; float y = 3; float z = 4; }
    payload = bytearray()
    payload += _encode_uint32(1, index)
    payload += _encode_float(2, x)
    payload += _encode_float(3, y)
    payload += _encode_float(4, z)
    return bytes(payload)


def _encode_landmark_set(set_type, landmarks):
    # message LandmarkSet { LandmarkSetType type = 1; repeated Landmark landmarks = 2; }
    payload = bytearray()
    payload += _encode_uint32(1, set_type)
    for index, point in enumerate(landmarks):
        payload += _encode_message(2, _encode_landmark(index, point[0], point[1], point[2]))
    return bytes(payload)


def encode_pose_frame_protobuf(frame_index, timestamp_ms, free_landmarks, anchored_landmarks):
    # message PoseFrame {
    #   uint64 frame_index = 1;
    #   uint64 timestamp_ms = 2;
    #   repeated LandmarkSet landmark_sets = 3;
    # }
    payload = bytearray()
    payload += _encode_uint64(1, frame_index)
    payload += _encode_uint64(2, timestamp_ms)
    payload += _encode_message(3, _encode_landmark_set(LANDMARK_SET_FREE, free_landmarks))
    payload += _encode_message(3, _encode_landmark_set(LANDMARK_SET_ANCHORED, anchored_landmarks))
    return bytes(payload)


class LowPassFilter:
    def __init__(self):
        self.initialized = False
        self.value = None

    def apply(self, value, alpha):
        if not self.initialized:
            self.value = value
            self.initialized = True
            return value

        self.value = alpha * value + (1.0 - alpha) * self.value
        return self.value


class OneEuroFilter:
    def __init__(self, min_cutoff, beta, derivative_cutoff):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.derivative_cutoff = derivative_cutoff
        self.value_filter = LowPassFilter()
        self.derivative_filter = LowPassFilter()
        self.last_value = None

    def alpha(self, cutoff, dt):
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def apply(self, value, dt):
        if dt <= 0:
            dt = 1.0 / max(global_vars.FPS, 1)

        derivative = 0.0 if self.last_value is None else (value - self.last_value) / dt
        smoothed_derivative = self.derivative_filter.apply(
            derivative,
            self.alpha(global_vars.SMOOTH_DERIVATIVE_CUTOFF, dt)
        )
        cutoff = self.min_cutoff + self.beta * abs(smoothed_derivative)
        smoothed_value = self.value_filter.apply(value, self.alpha(cutoff, dt))
        self.last_value = smoothed_value
        return smoothed_value


class LandmarkSmoother:
    def __init__(self, landmark_count=33, dimensions=3):
        self.filters = [
            [
                OneEuroFilter(
                    global_vars.SMOOTH_MIN_CUTOFF,
                    global_vars.SMOOTH_BETA,
                    global_vars.SMOOTH_DERIVATIVE_CUTOFF
                )
                for _ in range(dimensions)
            ]
            for _ in range(landmark_count)
        ]

    def apply(self, landmarks, dt):
        if not global_vars.SMOOTH_LANDMARKS:
            return landmarks

        smoothed = np.array(landmarks, dtype=np.float32, copy=True)
        for landmark_index in range(smoothed.shape[0]):
            for dimension in range(smoothed.shape[1]):
                smoothed[landmark_index][dimension] = self.filters[landmark_index][dimension].apply(
                    float(smoothed[landmark_index][dimension]),
                    dt
                )
        return smoothed

# the capture thread captures images from the WebCam on a separate thread (for performance)
class CaptureThread(threading.Thread):
    cap = None
    ret = None
    frame = None
    isRunning = False
    counter = 0
    timer = 0.0
    def run(self):
        self.cap = cv2.VideoCapture(global_vars.WEBCAM_INDEX) # sometimes it can take a while for certain video captures 4
        if global_vars.USE_CUSTOM_CAM_SETTINGS:
            self.cap.set(cv2.CAP_PROP_FPS, global_vars.FPS)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,global_vars.WIDTH)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT,global_vars.HEIGHT)

        time.sleep(1)
        
        print("Opened Capture @ %s fps"%str(self.cap.get(cv2.CAP_PROP_FPS)))
        self.timer = time.time()
        while not global_vars.KILL_THREADS:
            self.ret, self.frame = self.cap.read()
            self.isRunning = True
            if global_vars.DEBUG:
                self.counter = self.counter+1
                if time.time()-self.timer>=3:
                    print("Capture FPS: ",self.counter/(time.time()-self.timer))
                    self.counter = 0
                    self.timer = time.time()

# the body thread actually does the 
# processing of the captured images, and communication with unity
class BodyThread(threading.Thread):
    data = ""
    dirty = True
    pipe = None
    udp_socket = None
    udp_target = None
    timeSinceCheckedConnection = 0
    timeSincePostStatistics = 0
    lastPoseTime = None
    frameIndex = 0

    def __init__(self):
        super().__init__()
        self.init_udp_socket()

    def init_udp_socket(self):
        if not getattr(global_vars, "UDP_ENABLED", False):
            return

        self.udp_target = (
            getattr(global_vars, "UDP_TARGET_IP", "127.0.0.1"),
            int(getattr(global_vars, "UDP_TARGET_PORT", 9999))
        )
        try:
            self.udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            print("UDP pose output enabled: %s:%s" % self.udp_target)
        except OSError as ex:
            print("Failed to initialize UDP socket: %s" % ex)
            self.udp_socket = None
            self.udp_target = None

    def send_udp_pose_frame(self, payload):
        if self.udp_socket is None or self.udp_target is None or not payload:
            return

        try:
            self.udp_socket.sendto(payload, self.udp_target)
        except OSError as ex:
            print("Failed to send UDP pose frame: %s" % ex)

    def close_udp_socket(self):
        if self.udp_socket is not None:
            self.udp_socket.close()
            self.udp_socket = None

    def compute_real_world_landmarks(self,world_landmarks,image_landmarks,image_shape):
        try:
            # pseudo camera internals
            # if you properly calibrated your camera tracking quality can improve...
            frame_height,frame_width, channels = image_shape
            focal_length = frame_width*.6
            center = (frame_width/2, frame_height/2)
            camera_matrix = np.array(
                                    [[focal_length, 0, center[0]],
                                    [0, focal_length, center[1]],
                                    [0, 0, 1]], dtype = "double"
                                    )
            distortion = np.zeros((4, 1))

            success, rotation_vector, translation_vector = cv2.solvePnP(objectPoints= world_landmarks, 
                                                                        imagePoints= image_landmarks, 
                                                                        cameraMatrix= camera_matrix, 
                                                                        distCoeffs= distortion,
                                                                        flags=cv2.SOLVEPNP_SQPNP)
            transformation = np.eye(4)
            transformation[0:3, 3] = translation_vector.squeeze()

            # transform model coordinates into homogeneous coordinates
            model_points_hom = np.concatenate((world_landmarks, np.ones((33, 1))), axis=1)

            # apply the transformation
            world_points = model_points_hom.dot(np.linalg.inv(transformation).T)

            return world_points
        except AttributeError:
            print("Attribute Error: shouldn't happen frequently")
            return world_landmarks 

    def run(self):
        mp_drawing = mp.solutions.drawing_utils
        mp_pose = mp.solutions.pose
        free_smoother = LandmarkSmoother()
        anchored_smoother = LandmarkSmoother()
        
        capture = CaptureThread()
        capture.daemon = True
        capture.start()

        try:
            with mp_pose.Pose(min_detection_confidence=0.80, min_tracking_confidence=0.5, model_complexity = global_vars.MODEL_COMPLEXITY,static_image_mode = False,enable_segmentation = True) as pose: 
                
                while not global_vars.KILL_THREADS and capture.isRunning==False:
                    print("Waiting for camera and capture thread.")
                    time.sleep(0.5)
                print("Beginning capture")
                    
                while not global_vars.KILL_THREADS and capture.cap.isOpened():
                    ti = time.time()

                    # Fetch stuff from the capture thread
                    ret = capture.ret
                    image = capture.frame
                    if not ret or image is None:
                        time.sleep(0.01)
                        continue
                                    
                    # Image transformations and stuff
                    #image = cv2.flip(image, 1)
                    image.flags.writeable = global_vars.DEBUG
                    
                    # Detections
                    results = pose.process(image)
                    tf = time.time()
                    
                    # Rendering results
                    if global_vars.DEBUG:
                        if time.time()-self.timeSincePostStatistics>=1:
                            print("Theoretical Maximum FPS: %f"%(1/(tf-ti)))
                            self.timeSincePostStatistics = time.time()
                            
                        if results.pose_landmarks:
                            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS, 
                                                    mp_drawing.DrawingSpec(color=(255, 100, 0), thickness=2, circle_radius=4),
                                                    mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=2),
                                                    )
                        cv2.imshow('Body Tracking', image)
                        cv2.waitKey(1)

                    if self.pipe==None and time.time()-self.timeSinceCheckedConnection>=1:
                        try:
                            self.pipe = open(r'\\.\pipe\UnityMediaPipeBody', 'r+b', 0)
                        except (FileNotFoundError, OSError) as ex:
                            print("Waiting for Unity project to run... (%s)"%ex)
                            self.pipe = None
                        self.timeSinceCheckedConnection = time.time()

                    # Set up the current frame once, then fan it out to pipe and UDP.
                    self.data = ""
                    udp_payload = None
                    i = 0

                    if results.pose_world_landmarks and results.pose_landmarks:
                        current_pose_time = time.time()
                        if self.lastPoseTime is None:
                            pose_dt = 1.0 / max(capture.cap.get(cv2.CAP_PROP_FPS), global_vars.FPS, 1)
                        else:
                            pose_dt = current_pose_time - self.lastPoseTime
                        self.lastPoseTime = current_pose_time

                        image_landmarks = results.pose_landmarks
                        world_landmarks = results.pose_world_landmarks

                        model_points = np.float32([[-l.x, -l.y, -l.z] for l in world_landmarks.landmark])
                        image_points = np.float32([[l.x * image.shape[1], l.y * image.shape[0]] for l in image_landmarks.landmark])

                        body_world_landmarks_world = self.compute_real_world_landmarks(model_points,image_points,image.shape)
                        body_world_landmarks = results.pose_world_landmarks
                        anchored_landmarks = np.float32([[-l.x, -l.y, -l.z] for l in body_world_landmarks.landmark])

                        body_world_landmarks_world = free_smoother.apply(body_world_landmarks_world[:, :3], pose_dt)
                        anchored_landmarks = anchored_smoother.apply(anchored_landmarks, pose_dt)

                        for i in range(0,33):
                            self.data += "FREE|{}|{}|{}|{}\n".format(i,body_world_landmarks_world[i][0],body_world_landmarks_world[i][1],body_world_landmarks_world[i][2])
                        for i in range(0,33):
                            self.data += "ANCHORED|{}|{}|{}|{}\n".format(i,anchored_landmarks[i][0],anchored_landmarks[i][1],anchored_landmarks[i][2])

                        self.frameIndex += 1
                        udp_payload = encode_pose_frame_protobuf(
                            self.frameIndex,
                            int(current_pose_time * 1000),
                            body_world_landmarks_world,
                            anchored_landmarks
                        )

                    if self.pipe != None:
                        s = self.data.encode('utf-8') 
                        try:     
                            self.pipe.write(struct.pack('I', len(s)) + s)   
                            self.pipe.seek(0)    
                        except Exception as ex:  
                            print("Failed to write to pipe. Is the unity project open?")
                            try:
                                self.pipe.close()
                            except Exception:
                                pass
                            self.pipe= None

                    self.send_udp_pose_frame(udp_payload)
                            
                    #time.sleep(1/20)
        finally:
            global_vars.KILL_THREADS = True
            if self.pipe != None:
                self.pipe.close()
                self.pipe = None
            self.close_udp_socket()
            if capture.cap != None:
                capture.cap.release()
            cv2.destroyAllWindows()
