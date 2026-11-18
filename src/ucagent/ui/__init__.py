"""Interactive supervision page (Streamlit) on top of ``InteractiveSession``.

Only ``streamlit_app.py`` imports Streamlit; ``bootstrap.py`` (build the run
from the sidebar choices) and ``render.py`` (state to tables) are plain Python
so the page's logic is tested without a browser. Launch with ``ucagent ui``.
"""
