"""Offline sampling and conditional t2 budget audit; never loads device modules."""
from __future__ import annotations
import csv
import math
from collections import Counter
from pathlib import Path
import numpy as np

CONFIG = {"saved_nominal_fps": 10, "geometry_stride": 2,
          "timestamp_kind": "SDK_host_receive_not_exposure", "hardware_access": False,
          "material_speed_reference": None, "angle_reference": None, "t2": None}


def audit_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    t = np.array([float(r["host_time_s"]) for r in rows])
    frame = np.array([int(r["source_frame"]) for r in rows])
    ticks = np.array([int(r["device_ticks"]) for r in rows], dtype=np.int64)
    if len(t) < 2 or not np.isfinite(t).all() or np.any(np.diff(t) <= 0):
        raise ValueError("Invalid receive clock")
    dt = np.diff(t)
    return {"saved_frames": len(rows), "host_elapsed_s": float(t[-1]-t[0]),
            "saved_effective_fps": float((len(t)-1)/(t[-1]-t[0])),
            "dt_p50_ms": float(np.median(dt)*1000), "dt_p95_ms": float(np.percentile(dt,95)*1000),
            "dt_max_ms": float(np.max(dt)*1000),
            "sdk_frame_increment_counts": dict(Counter(map(str, np.diff(frame)))),
            "source_frame_nonincreasing_count": int(np.sum(np.diff(frame)<=0)),
            "device_tick_nonincreasing_count": int(np.sum(np.diff(ticks)<=0)),
            "device_tick_frequency_hz": None, "exposure_time_us": None,
            "gain": None, "actual_camera_acquisition_fps": None,
            "all_SDK_delivery_fps": None, "actual_hardware_ROI": None,
            "source_pixel_format": None, "camera_drop_count": None,
            "software_intentional_sampling": True, "geometry_effective_fps": float((len(t[::2])-1)/(t[::2][-1]-t[::2][0])),
            "saved_shape_wh": sorted(set((int(r["width"]),int(r["height"])) for r in rows)),
            "metric_velocity_m_s": None, "t2": None}


def conditional_period(radius, velocity, axial_pitch=0.0):
    """Synthetic sensitivity only, fixed-geometry no-slip helix, not production."""
    if not all(math.isfinite(x) for x in (radius,velocity,axial_pitch)) or radius<=0 or velocity<=0:
        raise ValueError("Positive finite radius and material velocity required")
    length = math.hypot(2*math.pi*radius, axial_pitch)
    return {"T_s": length/velocity, "dT_dR": 4*math.pi**2*radius/(velocity*length),
            "dT_dv": -length/velocity**2, "dT_dp": axial_pitch/(velocity*length),
            "status": "synthetic_or_conditional_not_measured", "t2_available": False}


def run(sources):
    return {"config":CONFIG, "sides":{side:audit_csv(Path(p).with_name(side+"_frames.csv")) for side,p in sources.items()},
            "implementation_findings":[
                "dual_camera_recorder._capture intentionally selects frames against target_fps=10 and received_at",
                "MvsCamera.open sets SDK latest-images-only and 3 buffer nodes; this is not a complete-delivery recorder",
                "MvsCamera.read stamps monotonic immediately after SDK GetImageBuffer; no calibrated exposure clock",
                "SDK converts source pixels to BGR8; recorder saves MJPG; historical conversion/compression effects unmeasured",
                "5Hz analysis is additional stride=2; no camera maximum frame rate can be inferred"],
            "unresolved":["device clock frequency and alignment","camera/SDK discarded frames","actual exposure and ROI","material speed and angle truth"],
            "t2_error_budget":{"status":"unavailable_no_numeric_physical_inputs",
              "primary":"independent t1 + signed continuous spool angle reaches 2*pi",
              "length_balance":"L_wound=integral(v_in dt)-[L_free(t)-L_free(t1)]",
              "conditional_length":"sqrt((2*pi*R_center)^2+p^2)",
              "timing_terms":["t1 interval","exposure/clock alignment","radius","material velocity","free-storage change","slip/stretch","entry azimuth/path","compute/control/actuator latency"],
              "material_identity_accuracy":None,"metric_velocity_m_s":None,"t2":None}}
