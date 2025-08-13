import streamlit as st
import numpy as np
import cv2
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from skimage import io, exposure, util, color, img_as_float
from scipy import ndimage
from streamlit_cropper import st_cropper
from PIL import Image

# --- Helper Function for Advanced Analysis ---
def perform_advanced_analysis(image):
    """Calculates center of mass and 1D profiles from a normalized 2D image."""
    cy, cx = ndimage.center_of_mass(image)
    horizontal_profile = np.sum(image, axis=0)  # profile vs. x
    vertical_profile = np.sum(image, axis=1)    # profile vs. y
    return (cx, cy), horizontal_profile, vertical_profile

# --- Main App ---
def main():
    st.set_page_config(layout="centered", page_title="Image Editor & Profiler")
    st.title("Image Editor & Density Profiler")
    st.write("Upload an image, crop it, adjust gamma, and analyze its density profile.")

    # Reset analysis state
    def reset_analysis():
        st.session_state.analysis_triggered = False

    # --- File Uploader ---
    uploaded_file = st.file_uploader(
        "Choose an image...",
        type=["jpg", "jpeg", "png"],
        on_change=reset_analysis
    )

    if uploaded_file is not None:
        # Load and preprocess image
        img_pil = Image.open(uploaded_file).convert("RGB")
        original_image = img_as_float(np.array(img_pil))

        st.sidebar.header("1. Crop & Adjust Image")
        
        # --- Crop using st_cropper ---
        st.sidebar.write("Crop the image interactively:")
        cropped_img = st_cropper(
            img_pil,
            realtime_update=True,
            box_color='#FF0000',
            aspect_ratio=None,
            return_type="image"
        )

        # Convert cropped PIL image to NumPy float
        cropped_np = img_as_float(np.array(cropped_img))

        # --- Gamma Correction ---
        gamma_val = st.sidebar.slider("Gamma Correction", 0.1, 5.0, 1.0)
        if gamma_val != 1.0:
            edited_image = exposure.adjust_gamma(cropped_np, gamma=gamma_val)
        else:
            edited_image = cropped_np

        # --- Display Original & Edited ---
        st.header("Edit and Preview")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original")
            st.image(original_image, use_container_width=True)
        with col2:
            st.subheader("Cropped & Adjusted")
            st.image(edited_image, use_container_width=True, clamp=True)

        st.markdown("---")

        # --- Analysis Trigger ---
        st.sidebar.header("2. Analyze Image")
        if st.sidebar.button("Analyze Edited Image"):
            st.session_state.analysis_triggered = True

        # --- Density Profiling ---
        if st.session_state.get('analysis_triggered', False):
            st.header("Density Profile Analysis")
            img_for_cv = (edited_image * 255).astype(np.uint8)
            gray_img = cv2.cvtColor(img_for_cv, cv2.COLOR_RGB2GRAY)

            # Resize & normalize
            resized = cv2.resize(gray_img, (200, 200))
            normalized = resized / 255.0

            # Analysis
            (center_x, center_y), h_profile, v_profile = perform_advanced_analysis(normalized)

            # Quantitative Metrics
            st.subheader("Quantitative Results")
            m1, m2 = st.columns(2)
            m1.metric("Center of Mass (X)", f"{center_x:.2f} px")
            m2.metric("Center of Mass (Y)", f"{center_y:.2f} px")
            st.markdown("---")

            # 2D Heatmap
            st.subheader("2D Heatmap View")
            show_com = st.checkbox("Show Center of Mass on 2D Map", value=True)
            fig_2d, ax2d = plt.subplots()
            im = ax2d.imshow(normalized, cmap='jet', origin='lower')
            if show_com:
                ax2d.plot(center_x, center_y, 'r+', markersize=12, label=f'Center of Mass ({center_x:.1f}, {center_y:.1f})')
                ax2d.legend()
            plt.colorbar(im, ax=ax2d, label='Normalized Density')
            ax2d.set_title("2D Density Profile")
            st.pyplot(fig_2d)

            # 1D Profiles
            st.subheader("1D Density Profiles")
            p1, p2 = st.columns(2)
            with p1:
                st.write("Horizontal Profile")
                st.line_chart(h_profile)
            with p2:
                st.write("Vertical Profile")
                st.line_chart(v_profile)

            # 3D Surface Plot
            st.subheader("Interactive 3D Density Profile")
            fig_3d = go.Figure(data=[go.Surface(z=normalized, colorscale='Jet')])
            fig_3d.update_layout(
                title='Atomic Cloud Density Surface',
                scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Density'),
                height=600,
                margin=dict(l=0, r=0, b=0, t=40)
            )
            st.plotly_chart(fig_3d, use_container_width=True)

    else:
        st.info("Awaiting image upload...")

if __name__ == "__main__":
    main()

