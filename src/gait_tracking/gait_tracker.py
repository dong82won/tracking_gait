import numpy as np
from scipy import signal
from matplotlib import pyplot as plt
import ahrs
from ahrs.common.orientation import q_conj, q_rot

def track_gait(time, gyro, acc, sample_period, start_time, stop_time):
    """IMU 데이터를 기반으로 3D 보행 궤적(Position)을 추정합니다."""

    # 1. 지정된 시간 구간 슬라이싱
    idx_sel = np.all([time >= start_time, time <= stop_time], axis=0)
    t = time[idx_sel]
    gyr = gyro[idx_sel]
    a = acc[idx_sel]
    fs = 1.0 / sample_period

    # 2. 정지 구간 탐지 (Stationary Phase Detection - SOS 적용)
    acc_mag = np.linalg.norm(a, axis=1)

    sos_hp = signal.butter(1, 0.001, btype='highpass', fs=fs, output='sos')
    acc_mag_filt = signal.sosfiltfilt(sos_hp, acc_mag)

    acc_mag_filt = np.abs(acc_mag_filt)

    sos_lp = signal.butter(1, 5.0, btype='lowpass', fs=fs, output='sos')
    acc_mag_filt = signal.sosfiltfilt(sos_lp, acc_mag_filt)

    stationary = acc_mag_filt < 0.05

    # 3. 자세 추정 (Mahony Filter)
    quat = np.zeros((t.size, 4), dtype=np.float64)
    # k_I=0.0 검증 통과를 위해 1e-9 전달
    mahony = ahrs.filters.Mahony(k_P=1.0, k_I=1e-9, frequency=fs)
    q = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

    init_period = 2.0
    init_idx = t <= t[0] + init_period
    init_acc = np.mean(a[init_idx], axis=0)
    for _ in range(2000):
        q = mahony.updateIMU(q, gyr=np.zeros(3), acc=init_acc)

    for i in range(t.size):
        mahony.k_P = 0.5 if stationary[i] else 0.0
        g_rad = gyr[i] * np.pi / 180.0
        quat[i, :] = mahony.updateIMU(q, gyr=g_rad, acc=a[i])

    # 4. 병진 가속도 계산 (중력 성분 제거)
    acc_earth = np.array([q_rot(q_conj(quat[i]), a[i]) for i in range(t.size)])
    acc_earth = (acc_earth - np.array([0, 0, 1])) * 9.81

    # 5. 속도 적분 및 ZUPT (Zero-Velocity Update)
    vel = np.zeros_like(acc_earth)
    for i in range(1, t.size):
        vel[i, :] = vel[i-1, :] + acc_earth[i, :] * sample_period
        if stationary[i]:
            vel[i, :] = np.zeros(3)

    # 6. 선형 드리프트 보정 (Integral Drift Correction)
    vel_drift = np.zeros_like(vel)
    stat_start = np.where(np.diff(stationary.astype(int)) == -1)[0] + 1
    stat_end = np.where(np.diff(stationary.astype(int)) == 1)[0] + 1

    for i in range(min(len(stat_start), len(stat_end))):
        start_idx, end_idx = stat_start[i], stat_end[i]
        if start_idx < end_idx:
            drift_rate = vel[end_idx - 1, :] / (end_idx - start_idx)
            enum = np.arange(end_idx - start_idx)[:, np.newaxis]
            vel_drift[start_idx:end_idx, :] = enum * drift_rate

    vel = vel - vel_drift

    # 7. 위치(Position) 적분
    pos = np.zeros_like(vel)
    for i in range(1, t.size):
        pos[i, :] = pos[i-1, :] + vel[i, :] * sample_period

    return pos

def plot_trajectory(pos):
    """3D 보행 궤적을 시각화합니다."""
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(pos[:, 0], pos[:, 1], pos[:, 2], label="Foot Trajectory", color='b')

    min_bound, max_bound = np.min(pos), np.max(pos)
    ax.set_xlim(min_bound, max_bound)
    ax.set_ylim(min_bound, max_bound)
    ax.set_zlim(min_bound, max_bound)

    ax.set_title("3D Foot Trajectory Tracking")
    ax.set_xlabel("X Position (m)")
    ax.set_ylabel("Y Position (m)")
    ax.set_zlabel("Z Position (m)")
    plt.legend()
    plt.show()
