from __future__ import annotations

import streamlit as st


main_page = st.Page(
    "app.py",
    title="YouTube Research Studio",
    icon=":material/query_stats:",
    url_path="main",
)


def redirect_to_main() -> None:
    st.switch_page(main_page)


root_page = st.Page(
    redirect_to_main,
    title="YouTube Research Studio",
    icon=":material/query_stats:",
    default=True,
    visibility="hidden",
)

current_page = st.navigation([root_page, main_page], position="hidden")
current_page.run()
