from pathlib import Path

from aind_behavior_services.rig import AindBehaviorRigModel
from aind_behavior_services.session import AindBehaviorSessionModel
from aind_behavior_services.task_logic import AindBehaviorTaskLogicModel

from contraqctor.contract import Dataset, DataStreamCollection
from contraqctor.contract.camera import Camera
from contraqctor.contract.csv import Csv
from contraqctor.contract.harp import (
    DeviceYmlByFile,
    HarpDevice,
)
from contraqctor.contract.json import PydanticModel, SoftwareEvents
from contraqctor.contract.mux import MapFromPaths
from contraqctor.contract.text import Text
from contraqctor.contract.utils import print_data_stream_tree

dataset_root = Path(r"path_to_data")
my_dataset = Dataset(
    name="my_dataset",
    version="1.0.0",
    description="My dataset",
    data_streams=[
        MapFromPaths(
            name="BehaviorVideos",
            description="Data from BehaviorVideos modality",
            reader_params=MapFromPaths.make_params(
                paths=dataset_root / "behavior-videos",
                include_glob_pattern=["*"],
                inner_data_stream=Camera,
                inner_param_factory=lambda x: Camera.make_params(path=dataset_root / "behavior-videos" / x),
            ),
        ),
        DataStreamCollection(
            name="Behavior",
            description="Data from the Behavior modality",
            data_streams=[
                HarpDevice(
                    name="HarpBehavior",
                    reader_params=HarpDevice.make_params(
                        path=dataset_root / "behavior/Behavior.harp",
                        device_yml_hint=DeviceYmlByFile(),
                    ),
                ),
                HarpDevice(
                    name="HarpManipulator",
                    reader_params=HarpDevice.make_params(
                        path=dataset_root / "behavior/StepperDriver.harp",
                        device_yml_hint=DeviceYmlByFile(),
                    ),
                ),
                HarpDevice(
                    name="HarpTreadmill",
                    reader_params=HarpDevice.make_params(
                        path=dataset_root / "behavior/Treadmill.harp",
                        device_yml_hint=DeviceYmlByFile(),
                    ),
                ),
                HarpDevice(
                    name="HarpOlfactometer",
                    reader_params=HarpDevice.make_params(
                        path=dataset_root / "behavior/Olfactometer.harp",
                        device_yml_hint=DeviceYmlByFile(),
                    ),
                ),
                HarpDevice(
                    name="HarpSniffDetector",
                    reader_params=HarpDevice.make_params(
                        path=dataset_root / "behavior/SniffDetector.harp",
                        device_yml_hint=DeviceYmlByFile(),
                    ),
                ),
                HarpDevice(
                    name="HarpLickometer",
                    reader_params=HarpDevice.make_params(
                        path=dataset_root / "behavior/Lickometer.harp",
                        device_yml_hint=DeviceYmlByFile(),
                    ),
                ),
                HarpDevice(
                    name="HarpClockGenerator",
                    reader_params=HarpDevice.make_params(
                        path=dataset_root / "behavior/ClockGenerator.harp",
                        device_yml_hint=DeviceYmlByFile(),
                    ),
                ),
                HarpDevice(
                    name="HarpEnvironmentSensor",
                    reader_params=HarpDevice.make_params(
                        path=dataset_root / "behavior/EnvironmentSensor.harp",
                        device_yml_hint=DeviceYmlByFile(),
                    ),
                ),
                DataStreamCollection(
                    name="HarpCommands",
                    description="Commands sent to Harp devices",
                    data_streams=[
                        HarpDevice(
                            name="HarpBehavior",
                            reader_params=HarpDevice.make_params(
                                path=dataset_root / "behavior/HarpCommands/Behavior.harp",
                                device_yml_hint=DeviceYmlByFile(),
                            ),
                        ),
                        HarpDevice(
                            name="HarpManipulator",
                            reader_params=HarpDevice.make_params(
                                path=dataset_root / "behavior/HarpCommands/StepperDriver.harp",
                                device_yml_hint=DeviceYmlByFile(),
                            ),
                        ),
                        HarpDevice(
                            name="HarpTreadmill",
                            reader_params=HarpDevice.make_params(
                                path=dataset_root / "behavior/HarpCommands/Treadmill.harp",
                                device_yml_hint=DeviceYmlByFile(),
                            ),
                        ),
                        HarpDevice(
                            name="HarpOlfactometer",
                            reader_params=HarpDevice.make_params(
                                path=dataset_root / "behavior/HarpCommands/Olfactometer.harp",
                                device_yml_hint=DeviceYmlByFile(),
                            ),
                        ),
                        HarpDevice(
                            name="HarpSniffDetector",
                            reader_params=HarpDevice.make_params(
                                path=dataset_root / "behavior/HarpCommands/SniffDetector.harp",
                                device_yml_hint=DeviceYmlByFile(),
                            ),
                        ),
                        HarpDevice(
                            name="HarpLickometer",
                            reader_params=HarpDevice.make_params(
                                path=dataset_root / "behavior/HarpCommands/Lickometer.harp",
                                device_yml_hint=DeviceYmlByFile(),
                            ),
                        ),
                        HarpDevice(
                            name="HarpClockGenerator",
                            reader_params=HarpDevice.make_params(
                                path=dataset_root / "behavior/HarpCommands/ClockGenerator.harp",
                                device_yml_hint=DeviceYmlByFile(),
                            ),
                        ),
                        HarpDevice(
                            name="HarpEnvironmentSensor",
                            reader_params=HarpDevice.make_params(
                                path=dataset_root / "behavior/HarpCommands/EnvironmentSensor.harp",
                                device_yml_hint=DeviceYmlByFile(),
                            ),
                        ),
                    ],
                ),
                DataStreamCollection(
                    name="SoftwareEvents",
                    description="Software events generated by the workflow. The timestamps of these events are low precision and should not be used to align to physiology data.",
                    data_streams=[
                        SoftwareEvents(
                            name="ActivePatch",
                            description="An event emitted when a patch threshold is crossed.",
                            reader_params=SoftwareEvents.make_params(
                                dataset_root / "behavior/SoftwareEvents/ActivePatch.json"
                            ),
                        ),
                        SoftwareEvents(
                            name="ActiveSite",
                            description="An event emitted when a site becomes active.",
                            reader_params=SoftwareEvents.make_params(
                                dataset_root / "behavior/SoftwareEvents/ActiveSite.json"
                            ),
                        ),
                        SoftwareEvents(
                            name="ArmOdor",
                            description="An event sent each time an Odor mixture messaged is sent to arm at the olfactometer.",
                            reader_params=SoftwareEvents.make_params(
                                dataset_root / "behavior/SoftwareEvents/ArmOdor.json"
                            ),
                        ),
                        SoftwareEvents(
                            name="Block",
                            description="An event signaling block transitions.",
                            reader_params=SoftwareEvents.make_params(
                                dataset_root / "behavior/SoftwareEvents/Block.json"
                            ),
                        ),
                        SoftwareEvents(
                            name="ChoiceFeedback",
                            description="A unit event that is emitted when the subject receives feedback about their choice.",
                            reader_params=SoftwareEvents.make_params(
                                dataset_root / "behavior/SoftwareEvents/ChoiceFeedback.json"
                            ),
                        ),
                        SoftwareEvents(
                            name="DepletionVariable",
                            description="The value of the variable used to determine the depletion state of the current patch.",
                            reader_params=SoftwareEvents.make_params(
                                dataset_root / "behavior/SoftwareEvents/DepletionVariable.json"
                            ),
                        ),
                        SoftwareEvents(
                            name="GiveReward",
                            description="The amount of rward given to a subject. The value can be null if no reward was given (P=0) or 0.0 if the reward was delivered but calculated to be 0.",
                            reader_params=SoftwareEvents.make_params(
                                dataset_root / "behavior/SoftwareEvents/GiveReward.json"
                            ),
                        ),
                        SoftwareEvents(
                            name="PatchRewardAmount",
                            description="Amount of reward available to be collected in the upcoming site.",
                            reader_params=SoftwareEvents.make_params(
                                dataset_root / "behavior/SoftwareEvents/PatchRewardAmount.json"
                            ),
                        ),
                        SoftwareEvents(
                            name="PatchRewardAvailable",
                            description="Amount of reward left in the patch.",
                            reader_params=SoftwareEvents.make_params(
                                dataset_root / "behavior/SoftwareEvents/PatchRewardAvailable.json"
                            ),
                        ),
                        SoftwareEvents(
                            name="PatchRewardProbability",
                            description="Probability of reward being available to be collected in the upcoming site.",
                            reader_params=SoftwareEvents.make_params(
                                dataset_root / "behavior/SoftwareEvents/PatchRewardProbability.json"
                            ),
                        ),
                        SoftwareEvents(
                            name="RngSeed",
                            description="The value of the random number generator seed.",
                            reader_params=SoftwareEvents.make_params(
                                dataset_root / "behavior/SoftwareEvents/RngSeed.json"
                            ),
                        ),
                        SoftwareEvents(
                            name="StopVelocityThreshold",
                            description="The velocity threshold used to determine if the subject is stopped or not. In cm/s.",
                            reader_params=SoftwareEvents.make_params(
                                dataset_root / "behavior/SoftwareEvents/StopVelocityTreshold.json"
                            ),
                        ),
                        SoftwareEvents(
                            name="VisualCorridorSpecs",
                            description="Specification of the visual corridor instantiated to be rendered.",
                            reader_params=SoftwareEvents.make_params(
                                dataset_root / "behavior/SoftwareEvents/VisualCorridorSpecs.json"
                            ),
                        ),
                        SoftwareEvents(
                            name="WaitRewardOutcome",
                            description="The outcome of the period between choice and reward delivery.",
                            reader_params=SoftwareEvents.make_params(
                                dataset_root / "behavior/SoftwareEvents/WaitRewardOutcome.json"
                            ),
                        ),
                        SoftwareEvents(
                            name="WaitLickOutcome",
                            description="The outcome of the period between reward availability and lick detection.",
                            reader_params=SoftwareEvents.make_params(
                                dataset_root / "behavior/SoftwareEvents/WaitLickOutcome.json"
                            ),
                        ),
                        SoftwareEvents(
                            name="UpdaterStopDurationOffset",
                            description="Metadata for the updater of the StopDurationOffset parameter.",
                            reader_params=SoftwareEvents.make_params(
                                dataset_root / "behavior/SoftwareEvents/UpdaterStopDurationOffset.json"
                            ),
                        ),
                        SoftwareEvents(
                            name="UpdaterStopVelocityThreshold",
                            description="Metadata for the updater of the StopVelocityThreshold parameter.",
                            reader_params=SoftwareEvents.make_params(
                                dataset_root / "behavior/SoftwareEvents/UpdaterStopVelocityThreshold.json"
                            ),
                        ),
                        SoftwareEvents(
                            name="UpdaterRewardDelayOffset",
                            description="Metadata for the updater of the RewardDelayOffset parameter.",
                            reader_params=SoftwareEvents.make_params(
                                dataset_root / "behavior/SoftwareEvents/UpdaterRewardDelayOffset.json"
                            ),
                        ),
                        SoftwareEvents(
                            name="HabituationRewardAvailable",
                            description="In the habituation task mode, this event will be emitted whenever a reward is available to be collected.",
                            reader_params=SoftwareEvents.make_params(
                                dataset_root / "behavior/SoftwareEvents/HabituationRewardAvailable.json"
                            ),
                        ),
                    ],
                ),
                Csv(
                    "CurrentPosition",
                    description="The position of the animal in VR coordinates (cm). The timestamp is derived from the encoder reading that gave rise to the position change.",
                    reader_params=Csv.make_params(
                        path=dataset_root / "behavior/OperationControl/CurrentPosition.csv",
                    ),
                ),
                Csv(
                    "IsStopped",
                    description="The result of the ongoing stop detection algorithm. The timestamp is derived from the encoder reading that gave rise to the position change.",
                    reader_params=Csv.make_params(
                        path=dataset_root / "behavior/OperationControl/IsStopped.csv",
                    ),
                ),
                Csv(
                    "Torque",
                    description="The torque instructed to be applied to the treadmill. Timestamps are software-derived, use the Harp device events for hardware timestamps.",
                    reader_params=Csv.make_params(
                        path=dataset_root / "behavior/OperationControl/CurrentPosition.csv",
                    ),
                ),
                Csv(
                    name="RendererSynchState",
                    description="Contains information that allows the post-hoc alignment of visual stimuli to the behavior data.",
                    reader_params=Csv.make_params(path=dataset_root / "behavior/Renderer/RendererSynchState.csv"),
                ),
                DataStreamCollection(
                    name="Logs",
                    data_streams=[
                        Text(
                            name="Launcher",
                            description="Contains the console log of the launcher process.",
                            reader_params=Text.make_params(
                                path=dataset_root / "behavior/Logs/launcher.log",
                            ),
                        ),
                        SoftwareEvents(
                            name="EndSession",
                            description="A file that determines the end of the session. If the file is empty, the session is still running or it was not closed properly.",
                            reader_params=SoftwareEvents.make_params(
                                path=dataset_root / "behavior/Logs/EndSession.json",
                            ),
                        ),
                    ],
                ),
                DataStreamCollection(
                    name="InputSchemas",
                    description="Configuration files for the behavior rig, task_logic and session.",
                    data_streams=[
                        PydanticModel(
                            name="Rig",
                            reader_params=PydanticModel.make_params(
                                model=AindBehaviorRigModel,
                                path=dataset_root / "behavior/Logs/rig_input.json",
                            ),
                        ),
                        PydanticModel(
                            name="TaskLogic",
                            reader_params=PydanticModel.make_params(
                                model=AindBehaviorTaskLogicModel,
                                path=dataset_root / "behavior/Logs/tasklogic_input.json",
                            ),
                        ),
                        PydanticModel(
                            name="Session",
                            reader_params=PydanticModel.make_params(
                                model=AindBehaviorSessionModel,
                                path=dataset_root / "behavior/Logs/session_input.json",
                            ),
                        ),
                    ],
                ),
            ],
        ),
    ],
)


if __name__ == "__main__":
    print(my_dataset.at("Behavior").at("HarpManipulator").load().at("WhoAmI").load().data)
    len([x for x in my_dataset if ((not x.is_collection) and isinstance(x, SoftwareEvents))])

    exc = my_dataset.load_all()

    for e in exc if exc is not None else []:
        print(f"Stream: {e[0]}")
        print(f"Exception: {e[1]}")
        print()

    print(my_dataset.at("Behavior").at("HarpBehavior").at("WhoAmI").read())

    print(my_dataset.at("Behavior").at("HarpCommands").at("HarpBehavior").at("OutputSet").read())
    print(my_dataset.at("Behavior").at("SoftwareEvents"))
    print(my_dataset.at("Behavior").at("SoftwareEvents").at("DepletionVariable").read())
    print(my_dataset.at("Behavior").at("SoftwareEvents").at("DepletionVariable"))

    print(my_dataset.at("Behavior").at("IsStopped").data)
    print(my_dataset.at("Behavior").at("RendererSynchState").data)

    print(my_dataset["Behavior"]["InputSchemas"]["Session"].data)

    path = ""
    child = my_dataset.at("Behavior").at("SoftwareEvents").at("DepletionVariable")
    while child.parent is not None:
        path = f"{child.name}:{path}"
        child = child.parent
    print(path)

    print(my_dataset.at("Behavior").at("HarpBehavior").device_reader)

    with open("my_dataset.md", "w", encoding="UTF-8") as f:
        f.write(print_data_stream_tree(my_dataset))

    print(my_dataset.at("Behavior").at("HarpBehavior").resolved_name)
