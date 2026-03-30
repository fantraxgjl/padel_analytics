""" Streamlit dashboard to interact with the data collected """

import datetime
import hashlib
import json
import numpy as np
import os
import re
import subprocess
import tempfile
import requests
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import supervision as sv
import pims

# Ensure required directories exist at startup
from config import RESULTS_DIR
for _d in (
    "./cache",
    "./weights/players_detection",
    "./weights/ball_detection",
    "./weights/players_keypoints_detection",
    "./weights/court_keypoints_detection",
    RESULTS_DIR,
):
    os.makedirs(_d, exist_ok=True)

from trackers import (
    Keypoint,
    Keypoints,
    PlayerTracker,
    PlayerKeypointsTracker,
    BallTracker,
    KeypointsTracker,
    TrackingRunner
)
from analytics import DataAnalytics
from analytics.data_analytics import zone_breakdown, partner_synchrony
from analytics.hit_detection import detect_hits
from analytics.shot_classifier import classify_hits
from analytics.rally_analysis import segment_rallies, enrich_rallies, analyse_rallies_with_claude
from visualizations.padel_court import padel_court_2d, padel_court_2d_heatmap, padel_court_2d_zones
from estimate_velocity import BallVelocityEstimator, ImpactType
from utils.video import save_video
from config import *

COLLECT_DATA = True

SPEED_ZONES = [
    ("Standing", 0, 2),
    ("Walking", 2, 7),
    ("Jogging", 7, 15),
    ("Running", 15, 20),
    ("Sprinting", 20, 9999),
]


@st.fragment
def velocity_estimator(video_info: sv.VideoInfo):

    frame_index = st.slider(
        "Frames",
        0,
        video_info.total_frames,
        1,
    )

    image = np.array(st.session_state["video"][frame_index])
    st.image(image)

    with st.form("choose-frames"):
        frame_index_t0 = st.number_input(
            "First frame: ",
            min_value=0,
            max_value=video_info.total_frames,
        )
        frame_index_t1 = st.number_input(
            "Second frame: ",
            min_value=1,
            max_value=video_info.total_frames,
        )
        impact_type_ch = st.radio(
            "Impact type: ",
            options=["Floor", "Player"],
        )
        get_Vz = st.radio(
            "Consider difference in ball altitude: ",
            options=[False, True]
        )

        estimate = st.form_submit_button("Calculate velocity")

    if estimate:

        assert frame_index_t0 < frame_index_t1

        if st.session_state["players_tracker"] is None:
            st.error("Data missing.")
        else:
            estimator = BallVelocityEstimator(
                source_video_fps=video_info.fps,
                players_detections=st.session_state["players_tracker"].results.predictions,
                ball_detections=st.session_state["ball_tracker"].results.predictions,
                keypoints_detections=st.session_state["keypoints_tracker"].results.predictions,
            )

            if impact_type_ch == "Floor":
                impact_type = ImpactType.FLOOR
            elif impact_type_ch == "Player":
                impact_type = ImpactType.RACKET

            ball_velocity_data, ball_velocity = estimator.estimate_velocity(
                frame_index_t0, frame_index_t1, impact_type, get_Vz=get_Vz,
            )
            st.write(ball_velocity)
            st.write("Velocity: ", ball_velocity.norm)
            st.image(ball_velocity_data.draw_velocity(st.session_state["video"]))
            padel_court = padel_court_2d()
            padel_court.add_trace(
                go.Scatter(
                    x=[
                        ball_velocity_data.position_t0_proj[0],
                        ball_velocity_data.position_t1_proj[0],
                    ],
                    y=[
                        ball_velocity_data.position_t0_proj[1]*-1,
                        ball_velocity_data.position_t1_proj[1]*-1,
                    ],
                    marker=dict(
                        size=10,
                        symbol="arrow-bar-up",
                        angleref="previous",
                    ),
                )
            )
            st.plotly_chart(padel_court)


# --- Session state initialisation ---
for _key in ("video", "df", "fixed_keypoints_detection",
             "players_keypoints_tracker", "players_tracker",
             "ball_tracker", "keypoints_tracker", "runner", "analytics"):
    if _key not in st.session_state:
        st.session_state[_key] = None

# ─────────────────────────────────────────────────────────────────────────────
st.title("Padel Analytics")

# --- Video history ---
_index_path = f"{RESULTS_DIR}/index.json"
_history = json.load(open(_index_path)) if os.path.exists(_index_path) else []

if _history:
    _options = {
        f"{e['processed_at'][:10]} — {e['url'][:70]}": e
        for e in reversed(_history)
    }
    _selected = st.selectbox(
        "Load a previously analysed video:",
        ["— analyse a new video —"] + list(_options.keys()),
    )
    if _selected != "— analyse a new video —":
        _entry = _options[_selected]
        _csv_path = f"{RESULTS_DIR}/{_entry['video_id']}.csv"
        _analytics_path = f"{RESULTS_DIR}/{_entry['video_id']}_analytics.json"
        if os.path.exists(_csv_path):
            st.session_state["df"] = pd.read_csv(_csv_path)
            if os.path.exists(_analytics_path):
                with open(_analytics_path) as _af:
                    st.session_state["analytics"] = json.load(_af)
            else:
                st.session_state["analytics"] = None
            st.success(f"Loaded results for {_entry['url'][:60]}")

# --- In-progress analysis indicator ---
_url_cache_path = "./cache/current_video_url.txt"
if os.path.exists(_url_cache_path):
    _in_progress_url = open(_url_cache_path).read().strip()
    _completed_urls = {e["url"] for e in _history}
    if _in_progress_url and _in_progress_url not in _completed_urls:
        st.warning(
            "**Analysis in progress or interrupted**\n\n"
            f"`{_in_progress_url[:100]}`\n\n"
            "Paste the URL below and click **Load Video** to resume — "
            "the download and any completed trackers will be skipped automatically."
        )

st.subheader("Load Video")
video_url = st.text_input(
    "Paste a Google Drive file link or direct video URL:",
    placeholder="https://drive.google.com/file/d/FILE_ID/view  or  https://example.com/video.mp4",
)
load_video = st.button("Load Video")

if load_video or st.session_state["video"] is not None:

    if load_video:
        if not video_url:
            st.error("Please enter a video URL.")
            st.stop()

        st.session_state["df"] = None

        _url_cache = "./cache/current_video_url.txt"
        _skip_download = (
            os.path.exists("tmp.mp4")
            and os.path.getsize("tmp.mp4") > 0
            and os.path.exists(_url_cache)
            and open(_url_cache).read().strip() == video_url.strip()
        )

        if _skip_download:
            st.info("Video already downloaded — resuming from cache.")
        else:
            with st.spinner("Downloading video..."):
                tmp_input_path = tempfile.mktemp(suffix=".mp4")

                gdrive_match = re.search(r"/file/d/([a-zA-Z0-9_-]+)", video_url)
                if gdrive_match or "drive.google.com" in video_url:
                    import gdown
                    file_id = gdrive_match.group(1) if gdrive_match else video_url
                    gdown.download(id=file_id, output=tmp_input_path, quiet=False)
                else:
                    response = requests.get(video_url, stream=True, timeout=300)
                    response.raise_for_status()
                    with open(tmp_input_path, "wb") as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)

            subprocess.run(
                ["ffmpeg", "-y", "-i", tmp_input_path, "-vcodec", "libx264", "tmp.mp4"],
                check=True,
            )
            os.unlink(tmp_input_path)
            with open(_url_cache, "w") as _f:
                _f.write(video_url.strip())

    if st.session_state["df"] is None:

        with st.status("Analysing video...", expanded=True) as _status:

            def progress_callback(step_name, current, total):
                if step_name == "done":
                    return
                label = step_name.replace("_", " ").title()
                _status.write(f"▶ {label}  (step {current + 1} / {total})")

            _status.write("Setting up trackers...")
            video_info = sv.VideoInfo.from_video_path(video_path="tmp.mp4")
            fps, w, h, total_frames = (
                video_info.fps,
                int(video_info.width),
                int(video_info.height),
                video_info.total_frames,
            )

            if FIXED_COURT_KEYPOINTS_LOAD_PATH is not None and os.path.exists(FIXED_COURT_KEYPOINTS_LOAD_PATH):
                with open(FIXED_COURT_KEYPOINTS_LOAD_PATH, "r") as f:
                    SELECTED_KEYPOINTS = json.load(f)
                fixed_keypoints = Keypoints(
                    [
                        Keypoint(
                            id=i,
                            xy=tuple(float(x) for x in v)
                        )
                        for i, v in enumerate(SELECTED_KEYPOINTS)
                    ]
                )
                keypoints_array = np.array(SELECTED_KEYPOINTS, dtype=np.int32)
                polygon_zone = sv.PolygonZone(
                    np.concatenate(
                        (
                            np.expand_dims(keypoints_array[0], axis=0),
                            np.expand_dims(keypoints_array[1], axis=0),
                            np.expand_dims(keypoints_array[-1], axis=0),
                            np.expand_dims(keypoints_array[-2], axis=0),
                        ),
                        axis=0
                    ),
                    frame_resolution_wh=(w, h),
                )
            else:
                # No fixed keypoints file — detect keypoints per-frame; use full-frame polygon
                fixed_keypoints = None
                polygon_zone = sv.PolygonZone(
                    np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.int32),
                    frame_resolution_wh=(w, h),
                )

            st.session_state["fixed_keypoints_detection"] = fixed_keypoints

            st.session_state["players_tracker"] = PlayerTracker(
                PLAYERS_TRACKER_MODEL,
                polygon_zone,
                batch_size=PLAYERS_TRACKER_BATCH_SIZE,
                annotator=PLAYERS_TRACKER_ANNOTATOR,
                show_confidence=True,
                load_path=PLAYERS_TRACKER_LOAD_PATH,
                save_path=PLAYERS_TRACKER_SAVE_PATH,
            )

            st.session_state["player_keypoints_tracker"] = PlayerKeypointsTracker(
                PLAYERS_KEYPOINTS_TRACKER_MODEL,
                train_image_size=PLAYERS_KEYPOINTS_TRACKER_TRAIN_IMAGE_SIZE,
                batch_size=PLAYERS_KEYPOINTS_TRACKER_BATCH_SIZE,
                load_path=PLAYERS_KEYPOINTS_TRACKER_LOAD_PATH,
                save_path=PLAYERS_KEYPOINTS_TRACKER_SAVE_PATH,
            )

            st.session_state["ball_tracker"] = BallTracker(
                BALL_TRACKER_MODEL,
                BALL_TRACKER_INPAINT_MODEL,
                batch_size=BALL_TRACKER_BATCH_SIZE,
                median_max_sample_num=BALL_TRACKER_MEDIAN_MAX_SAMPLE_NUM,
                median=None,
                load_path=BALL_TRACKER_LOAD_PATH,
                save_path=BALL_TRACKER_SAVE_PATH,
            )

            st.session_state["keypoints_tracker"] = KeypointsTracker(
                model_path=KEYPOINTS_TRACKER_MODEL,
                batch_size=KEYPOINTS_TRACKER_BATCH_SIZE,
                model_type=KEYPOINTS_TRACKER_MODEL_TYPE,
                fixed_keypoints_detection=st.session_state["fixed_keypoints_detection"],
                load_path=KEYPOINTS_TRACKER_LOAD_PATH,
                save_path=KEYPOINTS_TRACKER_SAVE_PATH,
            )

            runner = TrackingRunner(
                trackers=[
                    st.session_state["players_tracker"],
                    st.session_state["player_keypoints_tracker"],
                    st.session_state["ball_tracker"],
                    st.session_state["keypoints_tracker"],
                ],
                video_path="tmp.mp4",
                inference_path=OUTPUT_VIDEO_PATH,
                start=0,
                end=MAX_FRAMES,
                collect_data=COLLECT_DATA,
            )

            runner.run(progress_callback=progress_callback)

            st.session_state["runner"] = runner

            df = runner.data_analytics.into_dataframe(runner.video_info.fps)
            st.session_state["df"] = df

            # Persist results for future sessions
            video_id = hashlib.md5(video_url.encode()).hexdigest()[:12]
            df.to_csv(f"{RESULTS_DIR}/{video_id}.csv", index=False)

            # Pre-compute and cache analytics
            _analytics_cache: dict = {
                "zone_breakdown": {
                    str(pid): zone_breakdown(df, pid)
                    for pid in (1, 2, 3, 4)
                },
                "partner_synchrony": {
                    "1_2": {
                        k: v for k, v in partner_synchrony(df, 1, 2).items()
                        if k in ("vertical_sync", "horizontal_sync", "avg_formation_width")
                    },
                    "3_4": {
                        k: v for k, v in partner_synchrony(df, 3, 4).items()
                        if k in ("vertical_sync", "horizontal_sync", "avg_formation_width")
                    },
                },
            }

            # Phase B: hit detection + rally segmentation (only if ball data present)
            if "ball_x" in df.columns and df["ball_x"].notna().any():
                _hits = detect_hits(df)
                _hits = classify_hits(_hits, df)
                _rallies = segment_rallies(df)
                _rallies = enrich_rallies(_rallies, df, _hits)
                _analytics_cache["hits"] = _hits
                _analytics_cache["rallies"] = _rallies
            _analytics_path = f"{RESULTS_DIR}/{video_id}_analytics.json"
            with open(_analytics_path, "w") as _af:
                json.dump(_analytics_cache, _af, indent=2)
            st.session_state["analytics"] = _analytics_cache

            _history = json.load(open(_index_path)) if os.path.exists(_index_path) else []
            _history.append({
                "video_id": video_id,
                "url": video_url,
                "processed_at": datetime.datetime.now().isoformat(),
                "total_frames": runner.video_info.total_frames,
                "fps": runner.video_info.fps,
            })
            with open(_index_path, "w") as _f:
                json.dump(_history, _f, indent=2)

            _status.update(label="Analysis complete!", state="complete", expanded=False)

    st.session_state["video"] = pims.Video("tmp.mp4")
    st.subheader("Uploaded Video")
    st.video("tmp.mp4")

    estimate_velocity = st.checkbox("Calculate Ball Velocity")
    if estimate_velocity:
        st.write("Select a frame to calculate ball velocity:")
        velocity_estimator(st.session_state["runner"].video_info)

    if st.session_state["df"] is not None:
        df = st.session_state["df"]

        velocity_type_choice = st.radio(
            "Velocity component",
            ["Horizontal", "Vertical", "Absolute"],
        )
        velocity_type_mapper = {
            "Horizontal": "x",
            "Vertical": "y",
            "Absolute": "norm",
        }
        velocity_type = velocity_type_mapper[velocity_type_choice]

        # ── Match Summary Dashboard ──────────────────────────────────────────
        st.header("Match Summary")
        summary_cols = st.columns(4)
        for i, player_id in enumerate((1, 2, 3, 4)):
            dist = df[f"player{player_id}_distance"].sum()
            max_v = df[f"player{player_id}_V{velocity_type}4"].abs().max() * 3.6
            avg_v = df[f"player{player_id}_V{velocity_type}4"].abs().mean() * 3.6
            with summary_cols[i]:
                st.metric(f"Player {player_id}", f"{dist:.0f} m", help="Total distance covered")
                st.metric("Max speed", f"{max_v:.1f} km/h")
                st.metric("Avg speed", f"{avg_v:.1f} km/h")

        # ── Export ────────────────────────────────────────────────────────────
        st.subheader("Export")
        export_col1, export_col2 = st.columns(2)
        with export_col1:
            st.download_button(
                label="Download CSV",
                data=df.to_csv(index=False).encode(),
                file_name="padel_analytics.csv",
                mime="text/csv",
            )
        with export_col2:
            if os.path.exists(OUTPUT_VIDEO_PATH):
                with open(OUTPUT_VIDEO_PATH, "rb") as _vid_f:
                    st.download_button(
                        label="Download Annotated Video",
                        data=_vid_f.read(),
                        file_name="results.mp4",
                        mime="video/mp4",
                    )

        # ── Velocity over time ────────────────────────────────────────────────
        st.subheader("Players velocity as a function of time")
        fig_vel = go.Figure()
        for player_id in (1, 2, 3, 4):
            fig_vel.add_trace(
                go.Scatter(
                    x=df["time"],
                    y=np.abs(df[f"player{player_id}_V{velocity_type}4"].to_numpy()),
                    mode="lines",
                    name=f"Player {player_id}",
                ),
            )
        st.plotly_chart(fig_vel)

        # ── Speed Zone Distribution ───────────────────────────────────────────
        st.subheader("Speed Zone Distribution")
        fig_zones = go.Figure()
        zone_colors = ["#3B82F6", "#10B981", "#F59E0B", "#EF4444", "#8B5CF6"]
        for (zone_name, lo, hi), color in zip(SPEED_ZONES, zone_colors):
            pcts = []
            for player_id in (1, 2, 3, 4):
                speeds = df[f"player{player_id}_V{velocity_type}4"].abs() * 3.6
                n = max(len(speeds), 1)
                pcts.append(100 * ((speeds >= lo) & (speeds < hi)).sum() / n)
            fig_zones.add_trace(go.Bar(
                name=zone_name,
                x=[f"Player {i}" for i in (1, 2, 3, 4)],
                y=pcts,
                marker_color=color,
            ))
        fig_zones.update_layout(
            barmode="stack",
            yaxis_title="% of time",
            legend_title="Speed zone",
        )
        st.plotly_chart(fig_zones)

        # ── Position Analysis ─────────────────────────────────────────────────
        st.subheader("Analyse players position, velocity and acceleration")

        col1, col2 = st.columns((1, 1))

        with col1:
            player_choice = st.radio("Player: ", options=[1, 2, 3, 4])

        with col2:
            min_value = df[f"player{player_choice}_V{velocity_type}4"].abs().min()
            max_value = df[f"player{player_choice}_V{velocity_type}4"].abs().max()
            velocity_interval = st.slider(
                "Velocity Interval",
                min_value,
                max_value,
                (min_value, max_value),
            )

        df["QUERY_VELOCITY"] = df[f"player{player_choice}_V{velocity_type}4"].abs()
        min_choice = velocity_interval[0]
        max_choice = velocity_interval[1]
        df_scatter = df.query("@min_choice <= QUERY_VELOCITY <= @max_choice")

        tab_scatter, tab_heat = st.tabs(["Position Scatter", "Density Heatmap"])

        with tab_scatter:
            court_scatter = padel_court_2d()
            court_scatter.add_trace(
                go.Scatter(
                    x=df_scatter[f"player{player_choice}_x"],
                    y=df_scatter[f"player{player_choice}_y"] * -1,
                    mode="markers",
                    name=f"Player {player_choice}",
                    text=df_scatter[f"player{player_choice}_V{velocity_type}4"].abs() * 3.6,
                    marker=dict(
                        color=df_scatter[f"player{player_choice}_V{velocity_type}4"].abs() * 3.6,
                        size=12,
                        showscale=True,
                        colorscale="jet",
                        cmin=min_value * 3.6,
                        cmax=max_value * 3.6,
                    )
                )
            )
            st.plotly_chart(court_scatter)

        with tab_heat:
            heat_fig = padel_court_2d_heatmap(
                df_scatter[f"player{player_choice}_x"],
                df_scatter[f"player{player_choice}_y"] * -1,
            )
            st.plotly_chart(heat_fig)

        # ── Position over time ────────────────────────────────────────────────
        court_time = padel_court_2d()
        time_span = st.slider(
            "Time Interval",
            0.0,
            df["time"].max(),
        )
        df_time = df.query("time <= @time_span")
        court_time.add_trace(
            go.Scatter(
                x=df_time[f"player{player_choice}_x"],
                y=df_time[f"player{player_choice}_y"] * -1,
                mode="markers",
                name=f"Player {player_choice}",
                text=df_time[f"player{player_choice}_V{velocity_type}4"].abs() * 3.6,
                marker=dict(
                    color=df_time[f"player{player_choice}_V{velocity_type}4"].abs() * 3.6,
                    size=12,
                    showscale=True,
                    colorscale="jet",
                    cmin=min_value * 3.6,
                    cmax=max_value * 3.6,
                )
            )
        )
        st.plotly_chart(court_time)

        # ── Court Zone Breakdown ──────────────────────────────────────────────
        st.header("Court Zone Breakdown")
        st.caption(
            "Front: |y| < 3 m (net zone)  ·  "
            "Transition: 3–6 m  ·  "
            "Back: |y| > 6 m"
        )

        _zone_data = {
            pid: zone_breakdown(df, pid) for pid in (1, 2, 3, 4)
        }

        fig_zone = go.Figure()
        _zone_colors = {
            "front": "#22c55e",
            "transition": "#f59e0b",
            "back": "#ef4444",
        }
        for zone_key, color in _zone_colors.items():
            fig_zone.add_trace(go.Bar(
                name=zone_key.capitalize(),
                x=[f"Player {pid}" for pid in (1, 2, 3, 4)],
                y=[_zone_data[pid][zone_key] for pid in (1, 2, 3, 4)],
                marker_color=color,
                orientation="v",
            ))
        fig_zone.update_layout(
            barmode="stack",
            yaxis_title="% of time",
            legend_title="Zone",
            height=320,
        )
        st.plotly_chart(fig_zone, use_container_width=True)

        # Court diagram with zone lines + player density per-player
        st.subheader("Position heatmap with zone boundaries")
        _zone_cols = st.columns(4)
        for i, pid in enumerate((1, 2, 3, 4)):
            with _zone_cols[i]:
                st.caption(f"Player {pid}")
                _zheat = padel_court_2d_zones(width=220)
                _zheat.add_trace(
                    go.Histogram2dContour(
                        x=df[f"player{pid}_x"],
                        y=df[f"player{pid}_y"] * -1,
                        colorscale="Hot",
                        reversescale=True,
                        opacity=0.65,
                        showscale=False,
                        ncontours=15,
                    )
                )
                _zheat.update_layout(height=440, showlegend=False)
                st.plotly_chart(_zheat, use_container_width=False)

        # ── Partner Synchrony ─────────────────────────────────────────────────
        st.header("Partner Synchrony")
        st.caption(
            "Rolling Pearson correlation (60-frame window) between partners' "
            "velocity components. +1 = perfectly in sync, −1 = perfectly mirrored."
        )

        for pair_label, pa, pb in [("Team A (Players 1 & 2)", 1, 2), ("Team B (Players 3 & 4)", 3, 4)]:
            st.subheader(pair_label)
            _sync = partner_synchrony(df, pa, pb)

            _sc1, _sc2, _sc3 = st.columns(3)
            with _sc1:
                _v = _sync["vertical_sync"]
                st.metric("Vertical sync (Vy)", f"{_v:.2f}" if _v is not None else "N/A",
                          help="Mean rolling correlation of forward/backward velocity")
            with _sc2:
                _h = _sync["horizontal_sync"]
                st.metric("Horizontal sync (Vx)", f"{_h:.2f}" if _h is not None else "N/A",
                          help="Mean rolling correlation of side-to-side velocity")
            with _sc3:
                _w = _sync["avg_formation_width"]
                st.metric("Avg formation width", f"{_w:.1f} m" if _w is not None else "N/A",
                          help="Median lateral distance between partners")

            fig_sync = go.Figure()
            fig_sync.add_trace(go.Scatter(
                x=df["time"],
                y=_sync["rolling_vertical_sync"],
                mode="lines",
                name="Vertical sync (Vy)",
                line=dict(color="#6366f1"),
            ))
            fig_sync.add_trace(go.Scatter(
                x=df["time"],
                y=_sync["rolling_horizontal_sync"],
                mode="lines",
                name="Horizontal sync (Vx)",
                line=dict(color="#f43f5e"),
            ))
            fig_sync.update_layout(
                yaxis=dict(title="Pearson r", range=[-1.1, 1.1]),
                xaxis_title="Time (s)",
                height=280,
                legend=dict(orientation="h", y=1.1),
            )
            fig_sync.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.4)
            st.plotly_chart(fig_sync, use_container_width=True)

        # ── Hit Counting & Shot Types ─────────────────────────────────────────
        st.header("Hit Counting & Shot Types")

        _has_ball_data = "ball_x" in df.columns and df["ball_x"].notna().any()
        if not _has_ball_data:
            st.info(
                "Ball position data not available in this CSV. "
                "Re-run the video analysis pipeline to collect ball court projections."
            )
        else:
            _analytics = st.session_state.get("analytics") or {}
            _hits = _analytics.get("hits")
            if _hits is None:
                _hits = detect_hits(df)
                _hits = classify_hits(_hits, df)

            if not _hits:
                st.write("No hit events detected.")
            else:
                # Hit count per player
                st.subheader("Hits per player")
                _hit_counts = {pid: 0 for pid in (1, 2, 3, 4)}
                for h in _hits:
                    _hit_counts[h["player_id"]] += 1
                _hc_cols = st.columns(4)
                for i, pid in enumerate((1, 2, 3, 4)):
                    with _hc_cols[i]:
                        st.metric(f"Player {pid}", _hit_counts[pid], help="Detected hit events")

                # Shot type distribution
                st.subheader("Shot type distribution")
                _all_shot_types = sorted(set(h.get("shot_type", "unknown") for h in _hits))
                _sc_data: dict[str, list] = {stype: [0, 0, 0, 0] for stype in _all_shot_types}
                for h in _hits:
                    _sc_data[h.get("shot_type", "unknown")][h["player_id"] - 1] += 1

                fig_shots = go.Figure()
                _shot_palette = [
                    "#6366f1", "#22c55e", "#f59e0b", "#ef4444",
                    "#14b8a6", "#f97316", "#8b5cf6", "#ec4899",
                ]
                for si, shot_name in enumerate(_all_shot_types):
                    fig_shots.add_trace(go.Bar(
                        name=shot_name,
                        x=[f"Player {pid}" for pid in (1, 2, 3, 4)],
                        y=_sc_data[shot_name],
                        marker_color=_shot_palette[si % len(_shot_palette)],
                    ))
                fig_shots.update_layout(
                    barmode="stack",
                    yaxis_title="Hit count",
                    legend_title="Shot type",
                    height=320,
                )
                st.plotly_chart(fig_shots, use_container_width=True)

                # Hit timeline
                st.subheader("Hit timeline")
                fig_hits = go.Figure()
                for pid in (1, 2, 3, 4):
                    player_hits = [h for h in _hits if h["player_id"] == pid]
                    if player_hits:
                        fig_hits.add_trace(go.Scatter(
                            x=[h["time"] for h in player_hits],
                            y=[pid] * len(player_hits),
                            mode="markers",
                            name=f"Player {pid}",
                            marker=dict(size=8, symbol="circle"),
                            text=[h.get("shot_type", "") for h in player_hits],
                        ))
                fig_hits.update_layout(
                    yaxis=dict(
                        tickvals=[1, 2, 3, 4],
                        ticktext=["Player 1", "Player 2", "Player 3", "Player 4"],
                        title="",
                    ),
                    xaxis_title="Time (s)",
                    height=240,
                )
                st.plotly_chart(fig_hits, use_container_width=True)

        # ── AI Rally Analysis ─────────────────────────────────────────────────
        st.header("AI Rally Analysis")

        if not _has_ball_data:
            st.info("Requires ball position data — re-run pipeline first.")
        else:
            _analytics = st.session_state.get("analytics") or {}
            _cached_ai = _analytics.get("ai_analysis")

            if _cached_ai:
                _ai_result = _cached_ai
                st.caption("Loaded from cache.")
            else:
                _run_ai = st.button("Run AI Analysis", help="Calls Claude API — uses ANTHROPIC_API_KEY")
                _ai_result = None
                if _run_ai:
                    with st.spinner("Analysing with Claude..."):
                        _rallies_for_ai = _analytics.get("rallies") or []
                        _zone_for_ai = _analytics.get("zone_breakdown") or {
                            str(pid): zone_breakdown(df, pid) for pid in (1, 2, 3, 4)
                        }
                        _sync_for_ai = _analytics.get("partner_synchrony") or {
                            "1_2": {k: v for k, v in partner_synchrony(df, 1, 2).items()
                                    if k in ("vertical_sync", "horizontal_sync", "avg_formation_width")},
                            "3_4": {k: v for k, v in partner_synchrony(df, 3, 4).items()
                                    if k in ("vertical_sync", "horizontal_sync", "avg_formation_width")},
                        }
                        _ai_result = analyse_rallies_with_claude(
                            rallies=_rallies_for_ai,
                            zone_data=_zone_for_ai,
                            sync_data=_sync_for_ai,
                        )
                        # Cache to JSON
                        if "error" not in _ai_result:
                            _analytics["ai_analysis"] = _ai_result
                            _vid_id = hashlib.md5(
                                (st.session_state.get("current_video_url") or "").encode()
                            ).hexdigest()[:12]
                            _ai_json_path = f"{RESULTS_DIR}/{_vid_id}_analytics.json"
                            if os.path.exists(_ai_json_path):
                                with open(_ai_json_path) as _ajf:
                                    _existing = json.load(_ajf)
                                _existing["ai_analysis"] = _ai_result
                                with open(_ai_json_path, "w") as _ajf:
                                    json.dump(_existing, _ajf, indent=2)
                            st.session_state["analytics"] = _analytics

            if _ai_result:
                if "error" in _ai_result:
                    st.error(_ai_result["error"])
                else:
                    st.subheader("Per-player coaching feedback")
                    _fb_cols = st.columns(2)
                    _feedback = _ai_result.get("player_feedback", {})
                    for i, pid in enumerate((1, 2, 3, 4)):
                        with _fb_cols[i % 2]:
                            st.markdown(f"**Player {pid}**")
                            st.write(_feedback.get(str(pid), "No feedback available."))

                    st.subheader("Overall tactical patterns")
                    for pattern in _ai_result.get("overall_patterns", []):
                        st.markdown(f"- {pattern}")
