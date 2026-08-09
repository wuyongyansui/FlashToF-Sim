"""Run the user-edited configuration and print first-photon diagnostics."""

from example_config import USER_CONFIG
from flash_dtof.config import format_config
from flash_dtof.pipeline import format_diagnostics, run_simulation


def main():
    result = run_simulation(USER_CONFIG)
    print(format_config(result.user_config, result.derived_config))
    print()
    print(format_diagnostics(result))


if __name__ == "__main__":
    main()

