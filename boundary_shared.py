import pandas as pd
import streamlit as st

from xml_boundary import parse_course_limit_xml, parse_marks_xml


def sidebar_boundary_uploader() -> pd.DataFrame:
    st.markdown(
        "<div style='padding:8px;border-radius:8px;"
        "border:1px solid #7B1FA2;background:#F3E5F5;'>"
        "<b>XML commun – Boundary</b></div>",
        unsafe_allow_html=True,
    )

    xml_file = st.file_uploader(
        "Boundary XML (CourseLimit name='Boundary')",
        type=["xml"],
        accept_multiple_files=False,
    )

    if xml_file is None:
        # Keep keys clean when nothing is uploaded
        st.session_state.pop("marks_df", None)
        return pd.DataFrame(columns=["ring", "seq", "lat", "lon"])

    xml_bytes = xml_file.getvalue()

    # Boundary polygon
    boundary_df = parse_course_limit_xml(xml_bytes)

    # Also parse marks from the same XML, store in session_state for pages that need it
    try:
        marks_df = parse_marks_xml(xml_bytes)
    except Exception:
        marks_df = pd.DataFrame(columns=["mark", "lat", "lon"])
    st.session_state["marks_df"] = marks_df

    return boundary_df
