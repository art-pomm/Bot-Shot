import numpy as np
from pydrake.all import (
    AbstractValue,
    ConstantVectorSource,
    DiagramBuilder,
    LeafSystem,
    PiecewisePose,
    RigidTransform,
    RotationMatrix,
    Simulator,
    StartMeshcat,
)

from manipulation import running_as_notebook
from manipulation.exercises.grader import Grader
from manipulation.exercises.pick.test_robot_painter import TestRobotPainter
from manipulation.meshcat_utils import AddMeshcatTriad
from manipulation.scenarios import AddIiwaDifferentialIK
from manipulation.station import LoadScenario, MakeHardwareStation


# Start the visualizer.
meshcat = StartMeshcat()

class PoseTrajectorySource(LeafSystem):
    def __init__(self, pose_trajectory):
        LeafSystem.__init__(self)
        self._pose_trajectory = pose_trajectory
        self.DeclareAbstractOutputPort(
            "pose", lambda: AbstractValue.Make(RigidTransform()), self.CalcPose
        )

    def CalcPose(self, context, output):
        output.set_value(self._pose_trajectory.GetPose(context.get_time()))


class IIWA_Painter:
    def __init__(self, traj=None):
        builder = DiagramBuilder()
        scenario_data = """
        directives:
        - add_directives:
            file: package://manipulation/clutter.dmd.yaml
        model_drivers:
            iiwa: !IiwaDriver
                control_mode: position_only
                hand_model_name: wsg
            wsg: !SchunkWsgDriver {}
        """
        scenario = LoadScenario(data=scenario_data)
        self.station = builder.AddSystem(MakeHardwareStation(scenario, meshcat=meshcat))
        self.plant = self.station.GetSubsystemByName("plant")
        # Remove joint limits from the wrist joint.
        self.plant.GetJointByName("iiwa_joint_7").set_position_limits(
            [-np.inf], [np.inf]
        )
        controller_plant = self.station.GetSubsystemByName(
            "iiwa_controller_plant_pointer_system",
        ).get()

        # optionally add trajectory source
        if traj is not None:
            traj_source = builder.AddSystem(PoseTrajectorySource(traj))
            self.controller = AddIiwaDifferentialIK(
                builder,
                controller_plant,
                frame=controller_plant.GetFrameByName("body"),
            )
            builder.Connect(
                traj_source.get_output_port(),
                self.controller.get_input_port(0),
            )
            builder.Connect(
                self.station.GetOutputPort("iiwa.state_estimated"),
                self.controller.GetInputPort("robot_state"),
            )

            builder.Connect(
                self.controller.get_output_port(),
                self.station.GetInputPort("iiwa.position"),
            )
        else:
            iiwa_position = builder.AddSystem(ConstantVectorSource(np.zeros(7)))
            builder.Connect(
                iiwa_position.get_output_port(),
                self.station.GetInputPort("iiwa.position"),
            )

        wsg_position = builder.AddSystem(ConstantVectorSource([0.1]))
        builder.Connect(
            wsg_position.get_output_port(),
            self.station.GetInputPort("wsg.position"),
        )

        self.diagram = builder.Build()
        print(self.diagram)
        self.gripper_frame = self.plant.GetFrameByName("body")
        self.world_frame = self.plant.world_frame()

        context = self.CreateDefaultContext()
        self.diagram.ForcedPublish(context)

    def visualize_frame(self, name, X_WF, length=0.15, radius=0.006):
        """
        visualize imaginary frame that are not attached to existing bodies

        Input:
            name: the name of the frame (str)
            X_WF: a RigidTransform to from frame F to world.

        Frames whose names already exist will be overwritten by the new frame
        """
        AddMeshcatTriad(
            meshcat, "painter/" + name, length=length, radius=radius, X_PT=X_WF
        )

    def CreateDefaultContext(self):
        context = self.diagram.CreateDefaultContext()
        plant_context = self.diagram.GetMutableSubsystemContext(self.plant, context)

        # provide initial states
        q0 = np.array(
            [
                1.40666193e-05,
                1.56461165e-01,
                -3.82761069e-05,
                -1.32296976e00,
                -6.29097287e-06,
                1.61181157e00,
                -2.66900985e-05,
            ]
        )
        # set the joint positions of the kuka arm
        iiwa = self.plant.GetModelInstanceByName("iiwa")
        self.plant.SetPositions(plant_context, iiwa, q0)
        self.plant.SetVelocities(plant_context, iiwa, np.zeros(7))
        wsg = self.plant.GetModelInstanceByName("wsg")
        self.plant.SetPositions(plant_context, wsg, [-0.05, 0.05])
        self.plant.SetVelocities(plant_context, wsg, [0, 0])

        return context

    def get_X_WG(self, context=None):
        if not context:
            context = self.CreateDefaultContext()
        plant_context = self.plant.GetMyMutableContextFromRoot(context)
        X_WG = self.plant.CalcRelativeTransform(
            plant_context, frame_A=self.world_frame, frame_B=self.gripper_frame
        )
        return X_WG

    def paint(self, sim_duration=20.0):
        context = self.CreateDefaultContext()
        simulator = Simulator(self.diagram, context)

        meshcat.StartRecording(set_visualizations_while_recording=False)
        duration = sim_duration if running_as_notebook else 0.01
        simulator.AdvanceTo(duration)
        meshcat.PublishRecording()

        # define center and radius
radius = 0.1
p0 = [0.45, 0.0, 0.4]
R0 = RotationMatrix(np.array([[0, 1, 0], [0, 0, -1], [-1, 0, 0]]).T)
X_WCenter = RigidTransform(R0, p0)

num_key_frames = 10
"""
you may use different thetas as long as your trajectory starts
from the Start Frame above and your rotation is positive
in the world frame about +z axis
thetas = np.linspace(0, 2*np.pi, num_key_frames)
"""
thetas = np.linspace(0, 2 * np.pi, num_key_frames)

painter = IIWA_Painter()

X_WG = painter.get_X_WG()
painter.visualize_frame("gripper_current", X_WG)

RotationMatrix.MakeYRotation(np.pi / 6.0)

def compose_circular_key_frames(thetas, X_WCenter, radius):
    """
    returns: a list of RigidTransforms
    """
    # this is an template, replace your code below
    centerposition = X_WCenter.translation()

    key_frame_poses_in_world = []
    for theta in thetas:
        x = centerposition[0] + radius * np.cos(theta)
        y = centerposition[1] + radius * np.sin(theta)
        z = centerposition[2]

        xhat = centerposition[0] - x
        yhat = centerposition[1] - y
        vhat = np.array ([xhat, yhat, 0])
        vhat = vhat / np.linalg.norm(vhat)

        tangent = np.array([-np.sin(theta), np.cos(theta), 0])
        tangent = tangent / np.linalg.norm(tangent)

        cross = np.cross(vhat, tangent)
        cross = cross / np.linalg.norm(cross)

        zaxis = vhat
        xaxis = tangent
        yaxis = cross

        
        R_WG = RotationMatrix(np.vstack([xaxis, yaxis, zaxis]).T)

        this_pose = RigidTransform(R_WG, [x, y, z])
        key_frame_poses_in_world.append(this_pose)

    return key_frame_poses_in_world


def visualize_key_frames(frame_poses):
    for i, pose in enumerate(frame_poses):
        painter.visualize_frame("frame_{}".format(i), pose, length=0.05)


key_frame_poses = compose_circular_key_frames(thetas, X_WCenter, radius)
visualize_key_frames(key_frame_poses)

X_WGinit = painter.get_X_WG()
total_time = 1000
key_frame_poses = [X_WGinit] + compose_circular_key_frames(thetas, X_WCenter, radius)
times = np.linspace(0, total_time, num_key_frames + 1)
traj = PiecewisePose.MakeLinear(times, key_frame_poses)

painter = IIWA_Painter(traj)
painter.paint(sim_duration=total_time)