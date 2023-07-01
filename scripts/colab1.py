# NOTE: SETUP in  Colab

# !pip install -U pybullet numpngw

#!mkdir /home/annamalai
#%cd /home/annamalai
#!git clone https://github.com/RaMathuZen/geo_us_colab.git
#!mv geo_us_colab geo_us
#%cd geo_us/scripts/

# Script

from matplotlib import pylab
from numpngw import write_apng
import numpy as np
import pkgutil
import pybullet as p
import pybullet_data
import time
from IPython.display import Image,display, Javascript
import imageio_ffmpeg
import cv2 
from google.colab.patches import cv2_imshow
from google.colab.output import eval_js
from base64 import b64encode

# Ideas 
# 1. Implementing a Task Bar for changing the values of eye and target postion of camera , for zooming
# camera parameters
width = 640
height = 480
# egl = pkgutil.get_loader('eglRenderer')

fov = 90
aspect = width / height
near = 0.02
far = 10

flags = p.URDF_INITIALIZE_SAT_FEATURES
# view_matrix = p.computeViewMatrix([0, 1, 2.5], [0, 1, 0], [0, 1, 0]) #Top View
view_matrix = p.computeViewMatrix([1.5, 1.5, 1.5], [1.5, 0, 0], [0, -1, 1]) # Right Side View
projection_matrix = p.computeProjectionMatrixFOV(fov, aspect, near, far)

# vid = imageio_ffmpeg.write_frames('vid.mp4', (width,height), fps=240)
# vid.send(None) # seed the video writer with a blank frame


# frames=[]


class RobotController:
    def __init__(self, robot_type = 'ur5', controllable_joints = None, end_eff_index = None, time_step = 1e-1):
        global pub,pub_mes,of
        self.robot_type = robot_type
        self.robot_id = None
        self.num_joints = None
        self.controllable_joints = controllable_joints
        self.end_eff_index = end_eff_index
        self.time_step = time_step
        #of = DeviceFeedback()
        #of.lock = [False for i in range(3)]
        #pub = rospy.Publisher("/Geomagic/force_feedback",DeviceFeedback,queue_size=1)
        #pub_mes = rospy.Publisher("/ft_sensor/raw",Wrench,queue_size=1)


    # function to initiate pybullet and engine and create world
    def createWorld(self, GUI=True, view_world=False):
        global plugin
        # load pybullet physics engine
        #if GUI:
        #physicsClient = p.connect(p.GUI)
        #else:
        physicsClient = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        # plugin = p.loadPlugin(egl.get_filename(), "_eglRendererPlugin")
        # print("plugin=", plugin)
        #p.resetSimulation()
        p.resetSimulation(p.RESET_USE_DEFORMABLE_WORLD)
        GRAVITY = -9.8
        p.setGravity(0, 0, GRAVITY)
        p.setTimeStep(self.time_step)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI)
        p.setPhysicsEngineParameter(fixedTimeStep=self.time_step, numSolverIterations=100, numSubSteps=10,sparseSdfVoxelSize=0.25)
        p.setRealTimeSimulation(True)
        p.loadURDF("plane.urdf",flags=flags,useFixedBase=True)
        self.robot_id = p.loadURDF("/home/annamalai/geo_us/description/urdf/ur5.urdf",flags = flags, useFixedBase=True)

        self.num_joints = p.getNumJoints(self.robot_id) # Joints
        # print('#Joints:',self.num_joints)
        if self.controllable_joints is None:
            self.controllable_joints = list(range(1, self.num_joints-1))
        print('#Controllable Joints:', self.controllable_joints)
        if self.end_eff_index is None:
            self.end_eff_index = self.controllable_joints[-1]
        # print('#End-effector:', self.end_eff_index)
        p.enableJointForceTorqueSensor(self.robot_id,self.end_eff_index)
        if (view_world):
            while True:
                p.stepSimulation()
                time.sleep(self.time_step)

    # function for setting joint positions of robot
    def setJointPosition(self, position, kp=1.0, kv=1.0):
        # print('Joint position controller')
        zero_vec = [0.0] * len(self.controllable_joints)
        p.setJointMotorControlArray(self.robot_id,
                                    self.controllable_joints,
                                    p.POSITION_CONTROL,
                                    targetPositions=position,
                                    targetVelocities=zero_vec,
                                    positionGains=[kp] * len(self.controllable_joints),
                                    velocityGains=[kv] * len(self.controllable_joints))
        time1 = time.time()
        for i in range(1): #!!! to settle the robot to its position we can reduce it !!!
            p.stepSimulation()
            #view_matrix = p.computeViewMatrix([0, 1, 0.5], [0, 0, 0.5], [0, 1, 0])
            #projection_matrix = p.computeProjectionMatrixFOV(fov, width*1./height, near, far)
            image = p.getCameraImage(width, height, view_matrix,projection_matrix,shadow=True,lightDirection=[1, 1, 1])
            rgb_opengl = np.reshape(image[2], (height, width, 4)) * 1. / 255.
            np_img = np.reshape(image[2],(height,width,4))
            frame =  np_img[:,:,:3]
            # frames.append(frame)
            #print(np.size(image),np.size(rgb_opengl)) 
            #image_np = np.array(image, dtype=np.uint8).reshape((height,width, 4))[:, :, :3]
            # pylab.imshow(frame,interpolation='none', animated=True, label="pybullet")
            # vid.send(np.ascontiguousarray(frame))
            # vid.send(frame)
            bgr_image = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            # cv2_imshow(bgr_image)
            self.imshow("feed", bgr_image)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
        time2 = time.time()
        # print("Time taken for one step:",time2-time1)

    # function to solve inverse kinematics
    def solveInversePositionKinematics(self, end_eff_pose):
        # print('Inverse position kinematics')
        joint_angles =  p.calculateInverseKinematics(self.robot_id,self.end_eff_index,targetPosition=end_eff_pose[0:3],targetOrientation=p.getQuaternionFromEuler(end_eff_pose[3:6]))
        # print('Joint angles:', joint_angles)
        return joint_angles

    # function to get jacobian
    def getJacobian(self, joint_pos):
        eeState = p.getLinkState(self.robot_id, self.end_eff_index)
        link_trn, link_rot, com_trn, com_rot, frame_pos, frame_rot = eeState
        zero_vec = [0.0] * len(joint_pos)
        jac_t, jac_r = p.calculateJacobian(self.robot_id, self.end_eff_index, com_trn, list(joint_pos), zero_vec, zero_vec)
        J_t = np.asarray(jac_t)
        J_r = np.asarray(jac_r)
        J = np.concatenate((J_t, J_r), axis=0)
        # print('Jacobian:', J)
        return J
    
    def imshow(self, name, img):
        """Put frame as <img src="data:image/jpg;base64,...."> """

        js = Javascript('''
        async function showImage(name, image, width, height) {
          img = document.getElementById(name);
          if(img == null) {
            img = document.createElement('img');
            img.id = name;
            document.body.appendChild(img);
          }
          img.src = image;
          img.width = width;
          img.height = height;
        }
        ''')
        # print(name)
        height, width = img.shape[:2]

        ret, data = cv2.imencode('.jpg', img)   # compress array of pixels to JPG data
        data = b64encode(data)                  # encode base64
        data = data.decode()                    # convert bytes to string
        data = 'data:image/jpg;base64,' + data  # join header ("data:image/jpg;base64,") and base64 data (JPG)

        display(js)
        eval_js(f'showImage("{name}", "{data}", {width}, {height})')    

    #function to solve inverse velocity kinematics
    def solveInverseVelocityKinematics(self, end_eff_velocity):
        print('Inverse velocity kinematics')
        joint_pos, _ , _ = self.getJointStates()
        J  = self.getJacobian(joint_pos)
        if len(self.controllable_joints) > 1:
            joint_vel = np.linalg.pinv(J) @ end_eff_velocity
        else:
            joint_vel = J.T @ end_eff_velocity
        print('Joint velcoity:', joint_vel)
        return joint_vel


    # Function to define GUI sliders (name of the parameter,range,initial value)
    def TaskSpaceGUIcontrol(self, goal, max_limit = 3.14, min_limit = -3.14):
        xId = p.addUserDebugParameter("x", min_limit, max_limit, goal[0]) #x
        yId = p.addUserDebugParameter("y", min_limit, max_limit, goal[1]) #y
        zId = p.addUserDebugParameter("z", min_limit, max_limit, goal[2]) #z
        rollId = p.addUserDebugParameter("roll", min_limit, max_limit, goal[3]) #roll
        pitchId = p.addUserDebugParameter("pitch", min_limit, max_limit, goal[4]) #pitch
        yawId = p.addUserDebugParameter("yaw", min_limit, max_limit, goal[5]) # yaw
        return [xId, yId, zId, rollId, pitchId, yawId]

    def ForceGUIcontrol(self, forces, max_limit = 1.0, min_limit = -1.0):
        fxId = p.addUserDebugParameter("fx", min_limit, max_limit, forces[0]) #force along x
        fyId = p.addUserDebugParameter("fy", min_limit, max_limit, forces[1]) #force along y
        fzId = p.addUserDebugParameter("fz", min_limit, max_limit, forces[2]) #force along z
        mxId = p.addUserDebugParameter("mx", min_limit, max_limit, forces[3]) #moment along x
        myId = p.addUserDebugParameter("my", min_limit, max_limit, forces[4]) #moment along y
        mzId = p.addUserDebugParameter("mz", min_limit, max_limit, forces[5]) #moment along z
        return [fxId, fyId, fzId, mxId, myId, mzId]

    # function to read the value of task parameter
    def readGUIparams(self, ids):
        val1 = p.readUserDebugParameter(ids[0])
        val2 = p.readUserDebugParameter(ids[1])
        val3 = p.readUserDebugParameter(ids[2])
        val4 = p.readUserDebugParameter(ids[3])
        val5 = p.readUserDebugParameter(ids[4])
        val6 = p.readUserDebugParameter(ids[5])
        return np.array([val1, val2, val3, val4, val5, val6])

    # function to get desired joint trajectory
    def getTrajectory(self, thi, thf, tf, dt):
        desired_position, desired_velocity, desired_acceleration = [], [], []
        t = 0
        while t <= tf:
            th=thi+((thf-thi)/tf)*(t-(tf/(2*np.pi))*np.sin((2*np.pi/tf)*t))
            dth=((thf-thi)/tf)*(1-np.cos((2*np.pi/tf)*t))
            ddth=(2*np.pi*(thf-thi)/(tf*tf))*np.sin((2*np.pi/tf)*t)
            desired_position.append(th)
            desired_velocity.append(dth)
            desired_acceleration.append(ddth)
            t += dt
        desired_position = np.array(desired_position)
        desired_velocity = np.array(desired_velocity)
        desired_acceleration = np.array(desired_acceleration)
        return desired_position, desired_velocity, desired_acceleration

    def spawn_soft_ball(self):
        spinnerId = p.loadSoftBody("/home/annamalai/geo_us/description/urdf/ball.obj", simFileName = "/home/annamalai/geo_us/description/urdf/ball.vtk", basePosition = [0.4,0.4,0.5], scale = 0.25, mass = 1,useNeoHookean = 1, NeoHookeanMu = 400, NeoHookeanLambda = 600, NeoHookeanDamping = 0.05,useSelfCollision = 1, frictionCoeff = .5, collisionMargin = 0.001, useFaceContact=1)

robot = RobotController(robot_type='ur5')
robot.createWorld(GUI=True)
i = 0
robot.spawn_soft_ball()
# image = p.getCameraImage(width, height, view_matrix,projection_matrix,shadow=True,
#                           renderer=p.ER_BULLET_HARDWARE_OPENGL)
# rgb_opengl = np.reshape(image[2], (height, width, 4)) * 1. / 255.
# pylab.imshow(rgb_opengl,interpolation='none', animated=True, label="pybullet")
while (i < 0.45) :
    end_eff_pose = np.array([1, 0, 1-i, 0, 0, 0])
    joint_angles = robot.solveInversePositionKinematics(end_eff_pose)
    if joint_angles :
        robot.setJointPosition(joint_angles)
        end_last_pose = end_eff_pose
        last_joint_angle = joint_angles
    else :
        robot.setJointPosition(last_joint_angle)
    i = i + 0.05
    # print("Hi")
    #image = p.getCameraImage(width, height, view_matrix,projection_matrix,shadow=True,renderer=p.ER_BULLET_HARDWARE_OPENGL)
    #rgb_opengl = np.reshape(image[2], (height, width, 4)) * 1. / 255.
    # pylab.imshow(rgb_opengl,interpolation='none', animated=True, label="pybullet")

# vid.close()
# print(frames)
# %time write_apng("example.png",frames,delay=100)
# %time Image(filename="example.png")
# p.unloadPlugin(plugin)
# p.disconnect()


