import argparse
from gait_tracking import IMUDataLoader, track_gait, plot_trajectory

def main():
    parser = argparse.ArgumentParser(description="IMU ZUPT Gait Tracking")
    # 파일명은 사용자 환경에 맞게 입력 (확장자 이전 경로까지)
    parser.add_argument('--dataset', type=str, default='datasets/spiralStairs', help='Prefix of the dataset path')
    parser.add_argument('--start', type=float, default=4.0, help='Start time (seconds)')
    parser.add_argument('--stop', type=float, default=47.0, help='Stop time (seconds)')
    parser.add_argument('--freq', type=float, default=256.0, help='Sampling frequency (Hz)')

    args = parser.parse_args()
    sample_period = 1 / args.freq

    print(f"[*] 데이터를 불러오는 중: {args.dataset} ...")
    try:
        loader = IMUDataLoader(args.dataset, args.freq)
    except FileNotFoundError as e:
        print(f"[!] 에러: {e}")
        print("[!] 'datasets' 폴더에 해당 CSV 파일이 있는지 확인해주세요.")
        return

    print("[*] ZUPT 기반 보행 궤적 알고리즘을 처리하는 중 ...")

    position = track_gait(
        time=loader.time,
        gyro=loader.gyroscope,
        acc=loader.accelerometer,
        sample_period=sample_period,
        start_time=args.start,
        stop_time=args.stop
    )

    print("[*] 3D 궤적 렌더링 중 ... (창을 닫으면 프로그램이 종료됩니다)")
    plot_trajectory(position)

if __name__ == "__main__":
    main()
