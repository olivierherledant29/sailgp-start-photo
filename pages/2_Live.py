import streamlit as st
import time
from app_start_photo3 import render_start_photo

st.set_page_config(page_title="SailGP – Live", layout="wide")

REFRESH_S = 5  # secondes

# 1️⃣ Affichage normal
render_start_photo(
    page_title="SailGP – Live",
    mode_override="Live",
)

# 2️⃣ Footer discret
with st.empty():
    st.caption(f"Live refresh toutes les {REFRESH_S}s")

# 3️⃣ Refresh APRÈS affichage
time.sleep(REFRESH_S)
st.rerun()
