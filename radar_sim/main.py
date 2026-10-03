"""Entry point.   python main.py            live radar scope
                 python main.py --selfcheck   headless physics + DSP tests"""
import argparse
import sys


def main():
    ap = argparse.ArgumentParser(description="Simulated pulse radar with a live PPI / pulse-Doppler scope")
    ap.add_argument("--selfcheck", action="store_true", help="run the headless accuracy tests and exit")
    ap.add_argument("--seed", type=int, default=1, help="random seed (changes clutter map + noise)")
    args = ap.parse_args()

    if args.selfcheck:
        from radar.selfcheck import run
        sys.exit(0 if run() else 1)

    import matplotlib.pyplot as plt
    from radar.system import RadarSystem
    from radar.display import RadarApp
    app = RadarApp(RadarSystem(seed=args.seed))    # keep a reference so the animation isn't garbage collected
    plt.show()


if __name__ == "__main__":
    main()
