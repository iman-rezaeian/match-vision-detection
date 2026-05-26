"""Roster Manager — Streamlit page for managing players."""

import streamlit as st
import numpy as np
import os
import zipfile
import tempfile
from pathlib import Path
from PIL import Image

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import CUSTOM_CSS, DATA_DIR
from database.roster_db import RosterDB
try:
    from pipeline.face_reid import FaceReID
except ImportError:
    FaceReID = None


def get_db():
    if "roster_db" not in st.session_state:
        st.session_state["roster_db"] = RosterDB()
    return st.session_state["roster_db"]


st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
st.title("👥 Roster Manager")

db = get_db()

# --- Current Roster Table ---
st.header("Current Roster")

players = db.get_all_players()

if players:
    cols = st.columns([1, 3, 1, 2, 2, 2])
    cols[0].markdown("**Photo**")
    cols[1].markdown("**Name**")
    cols[2].markdown("**#**")
    cols[3].markdown("**Face ID**")
    cols[4].markdown("**Gait**")
    cols[5].markdown("**Cleat Color**")

    for player in players:
        cols = st.columns([1, 3, 1, 2, 2, 2])

        # Photo thumbnail
        if player["photo_path"] and os.path.exists(player["photo_path"]):
            try:
                img = Image.open(player["photo_path"])
                cols[0].image(img, width=40)
            except Exception:
                cols[0].markdown("📷")
        else:
            cols[0].markdown("—")

        cols[1].markdown(f"**{player['name']}**")
        cols[2].markdown(f"**{player['jersey_number']}**")

        # Face embedding status
        if player["face_embedding"]:
            cols[3].markdown("✅ Ready")
        elif player["photo_path"]:
            cols[3].markdown("⏳ Not processed")
        else:
            cols[3].markdown("— No photo")

        # Gait status
        if player["gait_signature"]:
            cols[4].markdown("✅ Ready")
        else:
            cols[4].markdown("⏳ Pending")

        # Cleat color
        if player["cleat_color_hsv"]:
            import json
            hsv = json.loads(player["cleat_color_hsv"])
            cols[5].markdown(f"🎨 H:{hsv['h']:.0f} S:{hsv['s']:.0f}")
        else:
            cols[5].markdown("—")
else:
    st.info("No players in roster. Add players below.")

st.divider()

# --- Add / Edit Player ---
st.header("Add / Edit Player")

with st.form("add_player_form"):
    col1, col2 = st.columns(2)
    with col1:
        name = st.text_input("Player Name")
        jersey = st.number_input("Jersey Number", min_value=1, max_value=99, step=1)
    with col2:
        hair_desc = st.text_input("Hair Description", placeholder="e.g. blonde short, dark curly")
        photo_file = st.file_uploader("Player Photo", type=["jpg", "jpeg", "png"])

    submitted = st.form_submit_button("💾 Save Player", use_container_width=True)

    if submitted:
        if not name:
            st.error("Player name is required.")
        else:
            # Save photo if uploaded
            photo_path = None
            if photo_file:
                photos_dir = DATA_DIR / "photos"
                photos_dir.mkdir(exist_ok=True)
                photo_path = str(photos_dir / f"{jersey}_{name.replace(' ', '_')}.jpg")
                with open(photo_path, "wb") as f:
                    f.write(photo_file.read())

            # Check if player exists (update) or new (insert)
            existing = db.get_player_by_jersey(jersey)
            if existing:
                updates = {"name": name, "hair_description": hair_desc}
                if photo_path:
                    updates["photo_path"] = photo_path
                db.update_player(existing["id"], **updates)
                st.success(f"Updated {name} (#{jersey})")
            else:
                db.add_player(name, jersey, photo_path, hair_desc)
                st.success(f"Added {name} (#{jersey})")

            # Extract face embedding if photo provided
            if photo_path:
                with st.spinner("Extracting face embedding..."):
                    try:
                        face_reid = FaceReID()
                        player = db.get_player_by_jersey(jersey)
                        if player:
                            face_reid.build_roster_embeddings([player], db)
                            st.success("✅ Face embedding extracted successfully!")
                    except Exception as e:
                        st.warning(f"Could not extract face embedding: {e}")

            st.rerun()

st.divider()

# --- Batch Import ---
st.header("Batch Import")
st.markdown("Upload a ZIP of photos named `{jersey_number}_{name}.jpg` to auto-import all players.")

zip_file = st.file_uploader("Upload ZIP file", type=["zip"])

if zip_file:
    if st.button("📦 Import All", use_container_width=True):
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Extract ZIP
            zip_path = os.path.join(tmp_dir, "roster.zip")
            with open(zip_path, "wb") as f:
                f.write(zip_file.read())

            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(tmp_dir)

            # Process photos
            photos_dir = DATA_DIR / "photos"
            photos_dir.mkdir(exist_ok=True)

            progress = st.progress(0)
            photo_files = [f for f in os.listdir(tmp_dir)
                           if f.lower().endswith(('.jpg', '.jpeg', '.png'))
                           and '_' in f]

            for i, filename in enumerate(photo_files):
                try:
                    parts = filename.rsplit('.', 1)[0].split('_', 1)
                    jersey_num = int(parts[0])
                    player_name = parts[1].replace('_', ' ')

                    # Copy photo
                    src = os.path.join(tmp_dir, filename)
                    dst = str(photos_dir / filename)

                    with open(src, "rb") as sf:
                        with open(dst, "wb") as df:
                            df.write(sf.read())

                    # Update or create player
                    existing = db.get_player_by_jersey(jersey_num)
                    if existing:
                        db.update_player(existing["id"], photo_path=dst)
                    else:
                        db.add_player(player_name, jersey_num, dst)

                    progress.progress((i + 1) / len(photo_files))

                except (ValueError, IndexError):
                    st.warning(f"Skipped {filename} — couldn't parse jersey_name format")

            st.success(f"Imported {len(photo_files)} player photos!")

            # Extract face embeddings for all
            with st.spinner("Extracting face embeddings for all players..."):
                try:
                    face_reid = FaceReID()
                    all_players = db.get_all_players()
                    face_reid.build_roster_embeddings(all_players, db)
                    st.success("✅ All face embeddings extracted!")
                except Exception as e:
                    st.warning(f"Some face extractions failed: {e}")

            st.rerun()

st.divider()

# --- Delete Player ---
st.header("Remove Player")
player_options = {f"{p['name']} (#{p['jersey_number']})": p['id'] for p in players}
if player_options:
    selected = st.selectbox("Select player to deactivate", options=list(player_options.keys()))
    if st.button("🗑️ Deactivate Player", type="secondary"):
        db.delete_player(player_options[selected])
        st.success(f"Deactivated {selected}")
        st.rerun()
