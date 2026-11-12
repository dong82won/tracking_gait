import numpy as np
from pathlib import Path

class IMUDataLoader:
    """IMU 데이터를 로드하고 파싱하는 클래스"""

    def __init__(self, file_prefix: str, sample_rate: float):
        self.file_prefix = file_prefix
        self.sample_rate = sample_rate
        self.file_appendage = '_CalInertialAndMag.csv'

        self.time = None
        self.gyroscope = None
        self.accelerometer = None
        self.magnetometer = None

        self._load_data()

    def _load_data(self):
        file_path = Path(f"{self.file_prefix}{self.file_appendage}")
        if not file_path.exists():
            raise FileNotFoundError(f"데이터 파일을 찾을 수 없습니다: {file_path}")

        try:
            # 첫 줄(헤더) 제외하고 데이터 로드
            data = np.loadtxt(file_path, delimiter=',', skiprows=1)
        except Exception as e:
            raise RuntimeError(f"CSV 데이터를 읽는 중 오류 발생: {e}")

        packet_num = data.shape[0]
        self.time = np.arange(0, packet_num) * (1 / self.sample_rate)

        # 센서별 데이터 슬라이싱
        self.gyroscope = data[:, 1:4]
        self.accelerometer = data[:, 4:7]
        self.magnetometer = data[:, 7:10]
