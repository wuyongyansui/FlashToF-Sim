"""正式的 NYU split 流式评估入口，不包含单 sample_id 路径。"""

import argparse
from dataclasses import replace

from example_config import CAMERA_CONFIG, NYU_BATCH_CONFIG, OUTPUT_CONFIG, SENSOR_CONFIG
from flash_dtof.batch import format_batch_summary, run_nyu_batch
from flash_dtof.output import (
    initialize_batch_output,
    save_batch_results,
    write_run_status,
)


def _parse_args():
    parser = argparse.ArgumentParser(
        description="让一个成对 NYU split 流式通过完整阵列仿真器。"
    )
    parser.add_argument(
        "--start",
        type=int,
        default=None,
        help="覆盖配置中的 split 零基起始索引",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="仅供开发的样本数量限制；省略则使用配置值或全量",
    )
    return parser.parse_args()


def _with_cli_overrides(batch_config, args):
    updates = {}
    if args.start is not None:
        updates["start"] = args.start
    if args.limit is not None:
        updates["limit"] = args.limit
    return replace(batch_config, **updates)


def main():
    args = _parse_args()
    batch_config = _with_cli_overrides(NYU_BATCH_CONFIG, args)
    run_directory = initialize_batch_output(
        OUTPUT_CONFIG,
        SENSOR_CONFIG,
        CAMERA_CONFIG,
        batch_config,
    )

    print("NYU FORMAL STREAMING BATCH")
    print("  dataset root                : {}".format(batch_config.dataset_root))
    print("  split                       : {}".format(batch_config.split))
    print("  start / limit               : {} / {}".format(
        batch_config.start,
        "all remaining" if batch_config.limit is None else batch_config.limit,
    ))
    print("  sensor [H,W,T] / periods    : [{},{},{}] / {}".format(
        SENSOR_CONFIG.image_height,
        SENSOR_CONFIG.image_width,
        SENSOR_CONFIG.num_time_bins,
        SENSOR_CONFIG.num_laser_periods,
    ))
    print("  memory policy               : one scene at a time")
    print("  output directory            : {}".format(run_directory))
    print()

    def report_progress(completed, total, metrics):
        if (
            completed == 1
            or completed == total
            or completed % 25 == 0
        ):
            print(
                "  [{:>4}/{}] {} detection={:.6f} RMSE={:.6f} m".format(
                    completed,
                    total,
                    metrics.sample_id,
                    metrics.detection_fraction,
                    metrics.root_mean_squared_error_m,
                )
            )

    try:
        summary = run_nyu_batch(
            SENSOR_CONFIG,
            batch_config,
            CAMERA_CONFIG,
            progress_callback=report_progress,
        )
        saved_paths = save_batch_results(run_directory, summary)
    except Exception as error:
        write_run_status(run_directory, "failed", error=error)
        raise
    print()
    print(format_batch_summary(summary))
    print("  summary JSON                : {}".format(saved_paths["summary_metrics"]))
    print("  per-scene CSV               : {}".format(saved_paths["scene_metrics"]))


if __name__ == "__main__":
    main()
