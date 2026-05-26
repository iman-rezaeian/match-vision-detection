"""Season Progress — Streamlit page for season trends."""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import CUSTOM_CSS, ACCENT_GREEN, TEAM_A_BLUE
from database.roster_db import RosterDB
from database.match_db import MatchDB


def get_db():
    if "roster_db" not in st.session_state:
        st.session_state["roster_db"] = RosterDB()
    return st.session_state["roster_db"]


def get_match_db():
    if "match_db" not in st.session_state:
        st.session_state["match_db"] = MatchDB()
    return st.session_state["match_db"]


st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
st.title("📈 Season Progress")

db = get_db()
match_db = get_match_db()

# --- Season Summary ---
st.header("Season Summary")

all_matches = match_db.get_all_matches()
all_stats = match_db.get_all_season_stats()

if not all_matches:
    st.info("No matches recorded yet. Analyze a match and save to season history to see progress here.")
    st.stop()

# Summary metrics
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Games Played", len(all_matches))
with col2:
    wins = sum(1 for m in all_matches if m.get("result") and "W" in m["result"].upper())
    losses = sum(1 for m in all_matches if m.get("result") and "L" in m["result"].upper())
    draws = len(all_matches) - wins - losses
    st.metric("Record", f"{wins}W-{draws}D-{losses}L")
with col3:
    if not all_stats.empty:
        total_dist = all_stats[all_stats["team"] == "Home"]["distance_m"].sum()
        st.metric("Total Team Distance", f"{total_dist/1000:.1f} km")
with col4:
    if not all_stats.empty:
        avg_goals = sum(1 for m in all_matches if m.get("result"))  # Placeholder
        st.metric("Avg ID Confidence", f"{all_stats['identification_confidence'].mean():.0%}")

st.divider()

# --- Player Progress Charts ---
st.header("Player Progress")

players = db.get_all_players()
player_options = {f"{p['name']} (#{p['jersey_number']})": p["id"] for p in players}

if player_options:
    selected_player_str = st.selectbox("Select Player", list(player_options.keys()))
    player_id = player_options[selected_player_str]

    player_stats = match_db.get_player_season_stats(player_id)

    if not player_stats.empty:
        # Line charts
        fig, axes = plt.subplots(2, 2, figsize=(14, 8))
        fig.patch.set_facecolor("#161b22")

        metrics = [
            ("distance_m", "Distance Covered (m)", axes[0, 0]),
            ("top_speed_ms", "Top Speed (m/s)", axes[0, 1]),
            ("sprints_count", "Sprint Count", axes[1, 0]),
            ("pct_att_third", "Time in Attacking Third (%)", axes[1, 1]),
        ]

        for col_name, title, ax in metrics:
            ax.set_facecolor("#161b22")
            if col_name in player_stats.columns:
                values = player_stats[col_name].values
                dates = range(1, len(values) + 1)

                ax.plot(dates, values, color=ACCENT_GREEN, linewidth=2, marker="o", markersize=6)
                ax.fill_between(dates, values, alpha=0.1, color=ACCENT_GREEN)

                # Trend line
                if len(values) >= 3:
                    z = np.polyfit(dates, values, 1)
                    p = np.poly1d(z)
                    ax.plot(dates, p(dates), color="#8b949e", linestyle="--", linewidth=1)

                    # Trend indicator
                    trend = "↗️" if z[0] > 0 else "↘️" if z[0] < 0 else "→"
                    ax.set_title(f"{title} {trend}", color="white", fontsize=11, fontweight="bold")
                else:
                    ax.set_title(title, color="white", fontsize=11, fontweight="bold")

                ax.set_xlabel("Game #", color="#8b949e", fontsize=9)
                ax.tick_params(colors="#8b949e")
                ax.spines["bottom"].set_color("#30363d")
                ax.spines["left"].set_color("#30363d")
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                ax.grid(True, alpha=0.1, color="#30363d")

        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.info(f"No season data yet for this player.")

st.divider()

# --- Team Trends ---
st.header("Team Trends")

if not all_stats.empty:
    home_stats = all_stats[all_stats["team"] == "Home"]

    if not home_stats.empty:
        # Average distance per game over season
        game_stats = home_stats.groupby("date").agg({
            "distance_m": "mean",
            "sprints_count": "mean",
            "pct_att_third": "mean",
        }).reset_index()

        if len(game_stats) >= 2:
            fig, axes = plt.subplots(1, 3, figsize=(14, 4))
            fig.patch.set_facecolor("#161b22")

            chart_data = [
                (game_stats["distance_m"], "Avg Distance/Player (m)", axes[0]),
                (game_stats["sprints_count"], "Avg Sprints/Player", axes[1]),
                (game_stats["pct_att_third"], "Avg Attacking Third %", axes[2]),
            ]

            for values, title, ax in chart_data:
                ax.set_facecolor("#161b22")
                games = range(1, len(values) + 1)
                ax.bar(games, values, color=TEAM_A_BLUE, alpha=0.8)
                ax.set_title(title, color="white", fontsize=11, fontweight="bold")
                ax.set_xlabel("Game #", color="#8b949e", fontsize=9)
                ax.tick_params(colors="#8b949e")
                ax.spines["bottom"].set_color("#30363d")
                ax.spines["left"].set_color("#30363d")
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)

            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

st.divider()

# --- Playing Time Tracker ---
st.header("Playing Time Tracker")

playing_time = match_db.get_playing_time_summary()

if not playing_time.empty:
    # Pivot table: players vs games
    pivot = playing_time.pivot_table(
        index="player_name", columns="date",
        values="minutes_played", aggfunc="sum"
    ).fillna(0)

    if not pivot.empty:
        fig, ax = plt.subplots(figsize=(14, max(6, len(pivot) * 0.4)))
        fig.patch.set_facecolor("#161b22")
        ax.set_facecolor("#161b22")

        # Stacked bar chart
        x = np.arange(len(pivot.columns))
        width = 0.8 / len(pivot.index)

        for i, player in enumerate(pivot.index):
            ax.barh(x + i * width, pivot.loc[player].values,
                    height=width, label=player, alpha=0.8)

        ax.set_yticks(x + width * len(pivot.index) / 2)
        ax.set_yticklabels(pivot.columns, color="#8b949e")
        ax.set_xlabel("Minutes Played", color="white")
        ax.set_title("Playing Time per Game", color="white",
                     fontsize=14, fontweight="bold")
        ax.tick_params(colors="#8b949e")
        ax.spines["bottom"].set_color("#30363d")
        ax.spines["left"].set_color("#30363d")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        ax.legend(loc="upper right", facecolor="#161b22", edgecolor="#30363d",
                  labelcolor="white", fontsize=7, ncol=2)

        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    # Total minutes summary
    st.subheader("Total Minutes Per Player")
    total_mins = playing_time.groupby("player_name")["minutes_played"].sum().sort_values(ascending=False)
    st.dataframe(
        pd.DataFrame({"Player": total_mins.index, "Total Minutes": total_mins.values.round(1)}),
        use_container_width=True, hide_index=True
    )
else:
    st.info("No playing time data yet.")

# --- Match History (with delete) ---
st.divider()
st.header("Match History")

for match in all_matches:
    mid = match["id"]
    date = match.get("date", "?")
    opp = match.get("opponent", "Unknown")
    res = match.get("result", "—")
    col_info, col_del = st.columns([5, 1])
    with col_info:
        st.write(f"**{date}** vs {opp} — {res}")
    with col_del:
        if st.button("🗑️", key=f"del_match_{mid}", help=f"Delete match {mid}"):
            match_db.delete_match(mid)
            st.success(f"Deleted match vs {opp} ({date})")
            st.rerun()
