from contract import my_dataset

import contraqctor.qc as qc
from contraqctor.contract.csv import Csv
from contraqctor.contract.harp import HarpDevice

harp_behavior: HarpDevice = my_dataset["Behavior"]["HarpBehavior"]
harp_sniff_detector: HarpDevice = my_dataset["Behavior"]["HarpSniffDetector"]
harp_behavior.load_all()
harp_sniff_detector.load_all()

clock_generator: HarpDevice = my_dataset["Behavior"]["HarpClockGenerator"]
clock_generator.load_all()
clock_generator_commands: HarpDevice = my_dataset["Behavior"]["HarpCommands"]["HarpClockGenerator"]
err = clock_generator_commands.load_all()

harp_behavior_commands: HarpDevice = my_dataset["Behavior"]["HarpCommands"]["HarpBehavior"]
harp_sniff_detector_commands: HarpDevice = my_dataset["Behavior"]["HarpCommands"]["HarpSniffDetector"]

is_stopped: Csv = my_dataset["Behavior"]["IsStopped"]
is_stopped.load_all()
harp_behavior_commands.load_all()
err3 = harp_sniff_detector_commands.load_all()

camera_test = my_dataset["BehaviorVideos"]
err2 = camera_test.load_all()

with qc.elevated_skips(False):
    runner = qc.Runner()
    runner.add_suite(qc.harp.HarpDeviceTestSuite(harp_behavior, harp_behavior_commands))
    runner.add_suite(qc.harp.HarpDeviceTestSuite(harp_sniff_detector, harp_sniff_detector_commands))
    runner.add_suite(qc.harp.HarpDeviceTestSuite(clock_generator, clock_generator_commands))
    runner.add_suite(qc.csv.CsvTestSuite(is_stopped))
    runner.add_suite(qc.harp.HarpHubTestSuite(clock_generator, [harp_behavior, harp_sniff_detector]))
    runner.add_suite(qc.harp.HarpSniffDetectorTestSuite(harp_sniff_detector))
    runner.add_suite(qc.camera.CameraTestSuite(camera_test["FaceCamera"]))
    runner.add_suite(qc.contract.ContractTestSuite(err + err2 + err3, exclude=[s for s, _ in err]))
    results = runner.run_all_with_progress(render_context=False)
