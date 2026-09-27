"""Streamlit visual styling."""

import streamlit as st


CUSTOM_CSS = """
<style>
    html, body, [class*="css"] {
        font-family: "Segoe UI", "Source Sans 3", sans-serif;
    }

    .block-container {
        padding-top: 1.4rem;
        max-width: 1200px;
    }

    .smartdoc-hero {
        background: linear-gradient(
            135deg,
            #0f3d4c 0%,
            #1a6a7a 55%,
            #2c8b7a 100%
        );
        color: #f4fbfa;
        padding: 1.4rem 1.6rem;
        border-radius: 16px;
        margin-bottom: 1.1rem;
        box-shadow: 0 10px 28px rgba(15, 61, 76, 0.28);
    }

    .smartdoc-hero h1 {
        margin: 0 0 0.35rem 0;
        font-size: 1.85rem;
        font-weight: 700;
    }

    .smartdoc-hero p {
        margin: 0;
        opacity: 0.95;
        line-height: 1.45;
    }

    .status-pill {
        display: inline-block;
        padding: 0.22rem 0.7rem;
        border-radius: 999px;
        font-size: 0.82rem;
        font-weight: 600;
        margin-top: 0.7rem;
        background: rgba(255, 255, 255, 0.16);
    }

    .source-card {
        border-left: 4px solid #1a6a7a;
        background: #f7fbfb;
        padding: 0.75rem 0.9rem;
        border-radius: 0 10px 10px 0;
        margin-bottom: 0.7rem;
    }

    .source-meta {
        color: #35565e;
        font-size: 0.86rem;
        margin-bottom: 0.35rem;
    }

    .warn-box,
    .err-box,
    .ok-box {
        padding: 0.8rem 1rem;
        border-radius: 10px;
        margin: 0.5rem 0 1rem 0;
    }

    .warn-box {
        background: #fff7e8;
        border: 1px solid #f0d29a;
    }

    .err-box {
        background: #fdeeee;
        border: 1px solid #e8b4b4;
    }

    .ok-box {
        background: #eef8f1;
        border: 1px solid #b7ddc3;
    }

    div[data-testid="stSidebar"] {
        background: #f3f8f8;
    }
</style>
"""


def apply_styles() -> None:
    """Apply custom CSS styles to the Streamlit application."""
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)