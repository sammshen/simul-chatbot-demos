import time
import os, sys
import numpy as np
import pandas as pd
import streamlit as st

# Prevent Streamlit file watcher from monitoring torch.classes
# This fixes the "no running event loop" error during hot-reloading
import streamlit.watcher.path_watcher
original_watch_file = streamlit.watcher.path_watcher.watch_file

def patched_watch_file(path, callback):
    if "torch/_classes.py" in path or "torch\\_classes.py" in path:
        return None
    return original_watch_file(path, callback)

streamlit.watcher.path_watcher.watch_file = patched_watch_file

from dual_chat_session import DualChatSession
import altair as alt
# Removing transformers import to avoid compatibility issues
import conversation_generator
import threading

# Set page configuration
st.set_page_config(
    page_title="LLM Performance Comparison",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# App title
st.title("Production Stack vs Other LLM Performance Comparison")
st.subheader("Comparing Time to First Token (TTFT) with real documentation")

# Note on how conversation history works
st.info("""
**Note on conversation management:**
This demo sends the same queries to both endpoints (Production Stack and Other) in parallel.
However, to ensure consistent context, only the responses from the Production Stack are added to the conversation history.
This prevents the conversation from branching with different responses from each model.
""")

# Initialize session state
if 'dual_session' not in st.session_state:
    with st.spinner("Loading documentation and initializing conversation history..."):
        st.session_state.dual_session = DualChatSession()
        st.session_state.history, st.session_state.docs = st.session_state.dual_session.initialize_with_conversation(5)
        st.session_state.ps_processing = False
        st.session_state.rs_processing = False
        st.session_state.metrics_history = []
        st.session_state.show_docs = False
        st.success("Initialization complete! Ready to start conversation.")
        # Force rerun to ensure UI is updated
        st.rerun()

# Sidebar with information
with st.sidebar:
    st.subheader("Documentation Loaded as Context")
    if st.session_state.docs:
        # Debug info about docs
        st.write(f"Loaded {len(st.session_state.docs)} documents:")

        # Display all document names
        for i, (doc_key, doc_name) in enumerate(st.session_state.docs.items()):
            # Simply display the full document name without any special formatting
            st.write(f"**Doc {i+1}**: {doc_name}")

        # Button to show/hide documentation previews
        if st.button("Show/Hide Documentation Previews"):
            st.session_state.show_docs = not st.session_state.show_docs

        if st.session_state.show_docs:
            # Create tabs for documentation
            doc_keys = list(st.session_state.docs.keys())
            doc_names = [f"Doc {i+1}" for i in range(len(doc_keys))]

            if doc_keys:  # Only create tabs if we have documents
                doc_tabs = st.tabs(doc_names)

                for i, (tab, doc_key) in enumerate(zip(doc_tabs, doc_keys)):
                    with tab:
                        doc_preview = st.session_state.dual_session.get_doc_preview(doc_key, 500)
                        st.text_area("Preview", doc_preview, height=200)

    # Add a reset button
    if st.button("Reset Conversation"):
        with st.spinner("Resetting conversation and reloading documentation..."):
            # Create new session objects
            st.session_state.dual_session = DualChatSession()
            st.session_state.history, st.session_state.docs = st.session_state.dual_session.initialize_with_conversation(5)
            st.session_state.ps_processing = False
            st.session_state.rs_processing = False
            st.session_state.metrics_history = []
            st.success("Conversation reset successfully!")
            st.rerun()

# Create two columns for the chat interfaces
col1, col2 = st.columns(2)

with col1:
    st.header("Production Stack", divider="gray")
    ps_container = st.container(height=600, border=False)

with col2:
    st.header("Other", divider="gray")
    rs_container = st.container(height=600, border=False)

# Display metrics
latest_ps_ttft = None
latest_rs_ttft = None

# Find the latest metrics for each endpoint
# Print all history items for debugging
print("All history items:")
for i, msg in enumerate(st.session_state.history):
    if msg.get("role") == "assistant":
        print(f"  {i}. {msg.get('endpoint', 'unknown')}: " +
              f"metrics={msg.get('metrics')}, " +
              f"content_len={len(msg.get('content', ''))}")

# Now search for metrics
for msg in reversed(st.session_state.history):
    if msg.get("role") == "assistant":
        if msg.get("endpoint") == "ProductionStack":
            metrics = msg.get("metrics", {})
            if metrics:
                if "ttft" in metrics and metrics["ttft"] is not None and latest_ps_ttft is None:
                    latest_ps_ttft = metrics["ttft"]
                    print(f"Found PS TTFT: {latest_ps_ttft}")

        elif msg.get("endpoint") == "RayServe" or msg.get("endpoint") == "Other":
            metrics = msg.get("metrics", {})
            if metrics:
                if "ttft" in metrics and metrics["ttft"] is not None and latest_rs_ttft is None:
                    latest_rs_ttft = metrics["ttft"]
                    print(f"Found Other TTFT: {latest_rs_ttft}")

    # Break when we have all metrics
    if (latest_ps_ttft is not None and latest_rs_ttft is not None):
        break

# Metrics comparison area in a more compact form
metrics_container = st.container(border=True)
with metrics_container:
    # Use smaller columns with a 3-column layout
    cols = st.columns([2, 2, 3])

    # Display TTFT metrics more compactly
    with cols[0]:
        st.markdown("**Production Stack TTFT**")
        if latest_ps_ttft is not None:
            st.markdown(f"<h4 style='margin:0'>{latest_ps_ttft:.2f} ms</h4>", unsafe_allow_html=True)
        else:
            st.markdown("<h4 style='margin:0'>N/A</h4>", unsafe_allow_html=True)

    with cols[1]:
        st.markdown("**Other TTFT**")
        if latest_rs_ttft is not None:
            st.markdown(f"<h4 style='margin:0'>{latest_rs_ttft:.2f} ms</h4>", unsafe_allow_html=True)
        else:
            st.markdown("<h4 style='margin:0'>N/A</h4>", unsafe_allow_html=True)

    # Display difference if both available
    with cols[2]:
        if latest_ps_ttft is not None and latest_rs_ttft is not None:
            diff = latest_ps_ttft - latest_rs_ttft
            if diff > 0:
                st.markdown(f"<b>Comparison:</b> Other is <b>{abs(diff):.2f} ms faster</b>", unsafe_allow_html=True)
            else:
                st.markdown(f"<b>Comparison:</b> Production Stack is <b>{abs(diff):.2f} ms faster</b>", unsafe_allow_html=True)

            # Display average if available
            metrics_comparison = st.session_state.dual_session.get_metrics_comparison()
            if metrics_comparison:
                st.markdown(f"<b>Average difference:</b> {metrics_comparison['avg_ttft_diff']:.2f} ms", unsafe_allow_html=True)
        else:
            st.markdown("<b>Waiting for metrics from both systems...</b>", unsafe_allow_html=True)

# Display the chat history
for i, message in enumerate(st.session_state.history):
    if message["role"] == "system":
        # For system messages, display them as notes at the top of both columns
        with ps_container:
            st.info(message["content"])
        with rs_container:
            st.info(message["content"])
    elif message["role"] == "user":
        with ps_container:
            st.chat_message("user").write(message["content"])
        with rs_container:
            st.chat_message("user").write(message["content"])

    elif message["role"] == "assistant":
        # Check if the message is simulated (part of the initial conversation)
        if message.get("simulated"):
            with ps_container:
                st.chat_message("assistant").write(message["content"])
            with rs_container:
                st.chat_message("assistant").write(message["content"])
        # Real responses from endpoints
        elif message.get("endpoint") == "ProductionStack":
            with ps_container:
                st.chat_message("assistant").write(message["content"])
        elif message.get("endpoint") == "RayServe" or message.get("endpoint") == "Other":
            with rs_container:
                st.chat_message("assistant").write(message["content"])

# Create an input field that spans both columns
prompt = st.chat_input("Type your question here")

def process_user_input(prompt):
    # Mark as processing
    st.session_state.ps_processing = True
    st.session_state.rs_processing = True

    # Get the streaming generators from both sessions
    ps_stream, rs_stream, ps_metrics, rs_metrics = st.session_state.dual_session.send_message(prompt)

    # Create placeholder messages for both endpoints
    with ps_container:
        ps_placeholder = st.chat_message("assistant").empty()

    with rs_container:
        rs_placeholder = st.chat_message("assistant").empty()

    # Initialize message strings
    ps_message = ""
    rs_message = ""

    # Create iterators from the generators
    ps_iterator = iter(ps_stream)
    rs_iterator = iter(rs_stream)

    # Track whether each stream is complete
    ps_complete = False
    rs_complete = False

    # Variables to control UI refresh rate
    update_frequency = 3  # Update UI every N tokens
    ps_update_counter = 0
    rs_update_counter = 0

    # Process both streams simultaneously by alternating
    while not (ps_complete and rs_complete):
        # Process tokens from Production Stack
        if not ps_complete:
            try:
                chunk = next(ps_iterator)
                if chunk.startswith("<metric:"):
                    print(f"Skipping PS metric chunk: {chunk}")
                    continue

                ps_message += chunk
                ps_update_counter += 1

                # Update UI at regular intervals
                if ps_update_counter >= update_frequency:
                    ps_placeholder.markdown(ps_message + "▌")
                    ps_update_counter = 0
            except StopIteration:
                ps_complete = True
                ps_placeholder.markdown(ps_message)
                st.session_state.ps_processing = False

        # Process tokens from Other
        if not rs_complete:
            try:
                chunk = next(rs_iterator)
                if chunk.startswith("<metric:"):
                    print(f"Skipping Other metric chunk: {chunk}")
                    continue

                rs_message += chunk
                rs_update_counter += 1

                # Update UI at regular intervals
                if rs_update_counter >= update_frequency:
                    rs_placeholder.markdown(rs_message + "▌")
                    rs_update_counter = 0
            except StopIteration:
                rs_complete = True
                rs_placeholder.markdown(rs_message)
                st.session_state.rs_processing = False

        # Small delay to allow other processes
        time.sleep(0.01)

    # Make sure final content is displayed
    if not ps_complete:
        ps_placeholder.markdown(ps_message)
    if not rs_complete:
        rs_placeholder.markdown(rs_message)

    # Print debug info about current metrics state
    print("Current metrics after processing:")
    for msg in reversed(st.session_state.history):
        if msg.get("role") == "assistant":
            metrics_debug = msg.get('metrics', {})
            print(f"  - {msg.get('endpoint', 'unknown')}: " +
                 f"TTFT = {metrics_debug.get('ttft')}")

    # Force a rerun to update metrics
    st.rerun()

if prompt and not st.session_state.ps_processing and not st.session_state.rs_processing:
    process_user_input(prompt)