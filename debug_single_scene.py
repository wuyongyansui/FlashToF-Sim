"""显式单场景调试入口，不属于正式 benchmark 路径。"""

import argparse

from example_config import NYU_BATCH_CONFIG, SENSOR_CONFIG
from flash_dtof.config import derive_config, format_config
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
    loader = NYUDepthV2Loader(
        NYU_BATCH_CONFIG.dataset_root,
        output_height=SENSOR_CONFIG.image_height,
        output_width=SENSOR_CONFIG.image_width,
        reflectivity_mode=NYU_BATCH_CONFIG.reflectivity_mode,
        constant_reflectivity=NYU_BATCH_CONFIG.constant_reflectivity,
    )
    loaded = loader.load(args.sample_id, split=args.split)
    result = run_simulation(SENSOR_CONFIG, loaded.scene_inputs)

    print("SINGLE-SCENE DEBUG ONLY (NOT A FORMAL BENCHMARK)")
    print("  dataset root                : {}".format(NYU_BATCH_CONFIG.dataset_root))
    print("  split / sample              : {} / {}".format(
        loaded.split, loaded.sample_id
    ))
    print("  source size [W,H]           : {}".format(loaded.source_size_wh))
    print("  center crop [L,T,R,B]       : {}".format(loaded.crop_box_ltrb))
    print("  output RGB / depth shapes   : {} / {}".format(
        loaded.rgb_u8_hwc.shape, loaded.scene_inputs.depth_m.shape
    ))
    print("  RGB / depth resize          : bilinear / nearest")
    print()
    print(format_config(SENSOR_CONFIG, derive_config(SENSOR_CONFIG), loaded.scene_inputs))
    print()
    print(format_diagnostics(result))


if __name__ == "__main__":
    main()
