import pybullet as p
import pybullet_data
import numpy as np
#import imageio_ffmpeg 
from numpngw import write_apng
from IPython.display import Image
from matplotlib import pylab
from google.colab import widgets
import time

# camera parameters
'''
cam_target_pos = [.95, 0, 0.2]
cam_distance = 2.05
cam_yaw, cam_pitch, cam_roll = -50, -40, 0
cam_width, cam_height = 480, 368
cam_up, cam_up_axis_idx, cam_near_plane, cam_far_plane, cam_fov = [0, 0, 1], 2, 0.01, 100, 60
vid = imageio_ffmpeg.write_frames('vid.mp4', (cam_width, cam_height), fps=240)
vid.send(None) # seed the video writer with a blank frame
'''

camTargetPos = [0, 0, 0]
cameraUp = [0, 0, 1]
cameraPos = [1, 1, 1]
frames=[]


class RobotController:
    def __init__(self, robot_type = 'ur5', controllable_joints = None, end_eff_index = None, time_step = 1e-2):
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
        # load pybullet physics engine
        #if GUI:
        #physicsClient = p.connect(p.GUI)
        #else:
        physicsClient = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        #p.resetSimulation()
        p.resetSimulation(p.RESET_USE_DEFORMABLE_WORLD)
        #spinnerId = p.loadSoftBody("/home/annamalai/geo_us/src/ros_geomagic/geomagic_description/urdf/ball.obj", simFileName = "/home/annamalai/geo_us/src/ros_geomagic/geomagic_description/urdf/ball.vtk", basePosition = [0,2,0.5], scale = 0.25, mass = 1,useNeoHookean = 1, NeoHookeanMu = 400, NeoHookeanLambda = 600, NeoHookeanDamping = 0.05,useSelfCollision = 1, frictionCoeff = .5, collisionMargin = 0.001, useFaceContact=1)
        GRAVITY = -9.8
        p.setGravity(0, 0, GRAVITY)
        p.setTimeStep(self.time_step)
        p.setPhysicsEngineParameter(fixedTimeStep=self.time_step, numSolverIterations=100, numSubSteps=10,sparseSdfVoxelSize=0.25)
        p.setRealTimeSimulation(True)
        p.loadURDF("plane.urdf")
        #p.loadURDF(fileName="/home/annamalai/geo_us/src/ros_geomagic/geomagic_description/urdf/cube.urdf",basePosition=[1,0,0],useFixedBase=True,globalScaling=1.5)
        #bunnyId = p.loadSoftBody("bunny.obj")
        #loading robot into the environment
        self.robot_id = p.loadURDF("/home/annamalai/geo_us/description/urdf/ur5.urdf", useFixedBase=True)

        self.num_joints = p.getNumJoints(self.robot_id) # Joints
        print('#Joints:',self.num_joints)
        if self.controllable_joints is None:
            self.controllable_joints = list(range(1, self.num_joints-1))
        print('#Controllable Joints:', self.controllable_joints)
        if self.end_eff_index is None:
            self.end_eff_index = self.controllable_joints[-1]
        print('#End-effector:', self.end_eff_index)
        p.enableJointForceTorqueSensor(self.robot_id,self.end_eff_index)
        if (view_world):
            while True:
                p.stepSimulation()
                time.sleep(self.time_step)

    # function for setting joint positions of robot
    def setJointPosition(self, position, kp=1.0, kv=1.0):
        print('Joint position controller')
        zero_vec = [0.0] * len(self.controllable_joints)
        p.setJointMotorControlArray(self.robot_id,
                                    self.controllable_joints,
                                    p.POSITION_CONTROL,
                                    targetPositions=position,
                                    targetVelocities=zero_vec,
                                    positionGains=[kp] * len(self.controllable_joints),
                                    velocityGains=[kv] * len(self.controllable_joints))
        for i in range(10): #!!! to settle the robot to its position we can reduce it !!!
            p.stepSimulation()
            cam_view_matrix = p.computeViewMatrixFromYawPitchRoll(cam_target_pos, cam_distance, cam_yaw, cam_pitch, cam_roll, cam_up_axis_idx)
            cam_projection_matrix = p.computeProjectionMatrixFOV(cam_fov, cam_width*1./cam_height, cam_near_plane, cam_far_plane)
            image = p.getCameraImage(cam_width, cam_height, cam_view_matrix, cam_projection_matrix)[2][:, :, :3]
            vid.send(np.ascontiguousarray(image))
   

    # function to solve inverse kinematics
    def solveInversePositionKinematics(self, end_eff_pose):
        print('Inverse position kinematics')
        joint_angles =  p.calculateInverseKinematics(self.robot_id,self.end_eff_index,targetPosition=end_eff_pose[0:3],targetOrientation=p.getQuaternionFromEuler(end_eff_pose[3:6]))
        print('Joint angles:', joint_angles)
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
        print('Jacobian:', J)
        return J

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
        spinnerId = p.loadSoftBody("/home/annamalai/geo_us/description/urdf/ball.obj", simFileName = "/home/annamalai/geo_us/src/description/urdf/ball.vtk", basePosition = [0,2,0.5], scale = 0.25, mass = 1,useNeoHookean = 1, NeoHookeanMu = 400, NeoHookeanLambda = 600, NeoHookeanDamping = 0.05,useSelfCollision = 1, frictionCoeff = .5, collisionMargin = 0.001, useFaceContact=1)

robot = RobotController(robot_type='ur5')
robot.createWorld(GUI=True)
i = 0
robot.spawn_soft_ball()
while (i > 0) :
    end_eff_pose = np.array([1, 0, 1-i, 0, 0, 0])
    joint_angles = robot.solveInversePositionKinematics(end_eff_pose)
    if joint_angles :
        robot.setJointPosition(joint_angles)
        end_last_pose = end_eff_pose
        last_joint_angle = joint_angles
    else :
        robot.setJointPosition(last_joint_angle)
    i = i + 0.01

vid.close()
p.disconnect()
