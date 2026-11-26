import sys
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

from ego_action.review import render_review

st.set_page_config(page_title="Episode review", layout="wide")
components.html(render_review(Path(sys.argv[1])), height=980, scrolling=True)
