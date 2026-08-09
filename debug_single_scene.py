"""显式单场景调试入口，不属于正式 benchmark 路径。"""

import argparse

from example_config import CAMERA_CONFIG, NYU_BATCH_CONFIG, OUTPUT_CONFIG, SENSOR_CONFIG
from flash_dtof.config import derive_config, format_config
from flash_dtof.geometry import load_nyu_rgb_intrinsics
from flash_dtof.irf import load_measured_irf
from flash_dtof.output import (
    initialize_debug_output,
    save_debug_results,
    write_run_status,
)
from flash_dtof.pipeline import format_diagnostics, run_simulation
from flash_dtof.scene import NYUDepthV2Loader


def _parse_args():
    parser = argparse.ArgumentParser(
        description="在正式批量入口之外调试一个指定的 NYU 场景。"
    )
    parser.add_argument("--sample-id", required=True, help="例如 nyu_0000")
    parser.add_argument(
        "--split",
        choices=("train", "val"),
        default=NYU_BATCH_CONFIG.split,
    )
    return parser.parse_args()


def main():
    args = _parse_args()
    run_directory = initialize_debug_output(
        OUTPUT_CONFIG,
        SENSOR_CONFIG,
        CAMERA_CONFIG,
        NYU_BATCH_CONFIG,
        args.split,
        args.sample_id,
    )
    hwt_bytes = (
        SENSOR_CONFIG.image_height
        * SENSOR_CONFIG.image_width
        * SENSOR_CONFIG.num_time_bins
        * 4
    )
    print("SINGLE-SCENE DEBUG OUTPUT POLICY")
    print("  output directory            : {}".format(run_directory))
    print("  save full EWH               : {}".format(OUTPUT_CONFIG.save_debug_ewh))
    if OUTPUT_CONFIG.save_debug_ewh:
        print("  EWH disk warning            : {:.2f} MiB / {:.2f} MB".format(
            hwt_bytes / (1024.0 ** 2), hwt_bytes / 1e6
        ))
    print("  transient + EWH core memory : at least {:.2f} MiB".format(
        2.0 * hwt_bytes / (1024.0 ** 2)
    ))
    print()
    loader = NYUDepthV2Loader(
        NYU_BATCH_CONFIG.dataset_root,
        expected_size_wh=(SENSOR_CONFIG.image_width, SENSOR_CONFIG.image_height),
        reflectivity_mode=NYU_BATCH_CONFIG.reflectivity_mode,
        constant_reflectivity=NYU_BATCH_CONFIG.constant_reflectivity,
    )
    try:
        camera_intrinsics = load_nyu_rgb_intrinsics(CAMERA_CONFIG, SENSOR_CONFIG)
        measured_irf = None
        if SENSOR_CONFIG.transient_model == "measured_irf":
            measured_irf = load_measured_irf(
                SENSOR_CONFIG.measured_irf_path,
                expected_bin_width_s=SENSOR_CONFIG.bin_width_s,
            )
        loaded = loader.load(args.sample_id, split=args.split)
        result = run_simulation(
            SENSOR_CONFIG,
            loaded.scene_inputs,
            camera_intrinsics,
            measured_irf=measured_irf,
        )
        saved_paths = save_debug_results(
            run_directory,
            loaded,
            result,
            OUTPUT_CONFIG,
        )
    except Exception as error:
        write_run_status(run_directory, "failed", error=error)
        raise

    print("SINGLE-SCENE DEBUG ONLY (NOT A FORMAL BENCHMARK)")
    print("  dataset root                : {}".format(NYU_BATCH_CONFIG.dataset_root))
    print("  split / sample              : {} / {}".format(
        loaded.split, loaded.sample_id
    ))
    print("  source size [W,H]           : {}".format(loaded.source_size_wh))
    print("  output RGB / depth-z shapes : {} / {}".format(
        loaded.rgb_u8_hwc.shape, loaded.scene_inputs.depth_z_m.shape
    ))
    print("  geometry transform          : native pixels; z -> slant via RGB rays")
    print("  output directory            : {}".format(run_directory))
    print()
    print(format_config(
        SENSOR_CONFIG,
        derive_config(SENSOR_CONFIG),
        loaded.scene_inputs,
        camera_intrinsics,
    ))
    print()
    print(format_diagnostics(result))
    print()
    print("SAVED DEBUG OUTPUTS")
    print("  diagnostics JSON            : {}".format(saved_paths["diagnostics"]))
    print("  NPY arrays                  : {} files".format(len(saved_paths["arrays"])))


if __name__ == "__main__":
    main()
