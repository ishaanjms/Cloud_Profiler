import streamlit as st
import numpy as np
import cv2
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from skimage import io, exposure, util, color
from scipy import ndimage

# --- Helper Function for Advanced Analysis ---
def perform_advanced_analysis(image):
    """Calculates center of mass and 1D profiles from a normalized 2D image."""
    # The result is in (row, column) format, which corresponds to (y, x)
    cy, cx = ndimage.center_of_mass(image)

    # Get 1D profiles by summing along axes. This integrates the signal.
    horizontal_profile = np.sum(image, axis=0) # Sum over rows to get profile vs. x
    vertical_profile = np.sum(image, axis=1)   # Sum over columns to get profile vs. y
    
    return (cx, cy), horizontal_profile, vertical_profile

# --- Main App ---
def main():
    st.set_page_config(layout="wide", page_title="Image Editor & Profiler")
    st.title("🔬 Image Editor & Density Profiler")
    st.write("Upload an image, edit it using the sidebar controls, and then click 'Analyze' to generate its density profile.")

    # --- State Management ---
    def reset_analysis():
        if 'analysis_triggered' in st.session_state:
            st.session_state.analysis_triggered = False

    # --- File Uploader ---
    uploaded_file = st.file_uploader(
        "Choose an image...",
        type=["jpg", "jpeg", "png"],
        on_change=reset_analysis
    )

    if uploaded_file is not None:
        # Load the uploaded image
        original_image = io.imread(uploaded_file)
        
        # Pre-process: Ensure image is float RGB
        if original_image.shape[-1] == 4: # Handle RGBA
            original_image = color.rgba2rgb(original_image)
        original_image = util.img_as_float(original_image)
        
        img_height, img_width, _ = original_image.shape

        # --- Sidebar Controls ---
        st.sidebar.header("1. Edit Image")
        
        # Cropping Controls
        st.sidebar.subheader("Crop")
        crop_left = st.sidebar.slider("Left (X)", 0, img_width - 1, 0)
        crop_top = st.sidebar.slider("Top (Y)", 0, img_height - 1, 0)
        crop_width = st.sidebar.slider("Width", 1, img_width, img_width)
        crop_height = st.sidebar.slider("Height", 1, img_height, img_height)
        
        st.sidebar.markdown("---")
        
        # Gamma Control
        st.sidebar.subheader("Tone")
        gamma_val = st.sidebar.slider("Gamma Correction", 0.1, 5.0, 1.0)

        # --- Image Editing Logic ---
        # Validate crop dimensions
        if crop_left + crop_width > img_width:
            crop_width = img_width - crop_left
        if crop_top + crop_height > img_height:
            crop_height = img_height - crop_top
        
        # Apply crop
        edited_image = original_image[crop_top : crop_top + crop_height, crop_left : crop_left + crop_width]
        
        # Apply gamma
        if gamma_val != 1.0 and edited_image.size > 0:
            edited_image = exposure.adjust_gamma(edited_image, gamma=gamma_val)
        
        # --- Display Editing View ---
        st.header("Edit and Preview")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original")
            st.image(original_image, use_container_width=True)
        with col2:
            st.subheader("Edited")
            if edited_image.size > 0:
                st.image(edited_image, use_container_width=True, clamp=True)

        st.markdown("---")

        # --- Analysis Trigger ---
        st.sidebar.header("2. Analyze Image")
        if st.sidebar.button("Analyze Edited Image"):
            st.session_state.analysis_triggered = True

        # --- Density Profiling Logic ---
        if st.session_state.get('analysis_triggered', False):
            st.header("Density Profile Analysis")
            
            if edited_image.size > 0:
                # Convert for OpenCV: float [0,1] RGB -> uint8 [0,255] BGR
                img_for_cv = (edited_image * 255).astype(np.uint8)
                gray_img = cv2.cvtColor(img_for_cv, cv2.COLOR_RGB2GRAY)
                
                # Resize and normalize
                resized = cv2.resize(gray_img, (200, 200))
                normalized = resized / 255.0

                # --- Call the new analysis function ---
                (center_x, center_y), h_profile, v_profile = perform_advanced_analysis(normalized)

                # --- Display Quantitative Metrics ---
                st.subheader("📊 Quantitative Results")
                metric_col1, metric_col2 = st.columns(2)
                metric_col1.metric("Center of Mass (X)", f"{center_x:.2f} px")
                metric_col2.metric("Center of Mass (Y)", f"{center_y:.2f} px")
                st.markdown("---")

                # --- Display 2D Heatmap with Center of Mass ---
                st.subheader("2D Heatmap View")
                
                # **NEW**: Add a checkbox to control the center of mass visibility
                show_com = st.checkbox("Show Center of Mass on 2D Map", value=True)
                
                fig_2d, ax2d = plt.subplots()
                im = ax2d.imshow(normalized, cmap='jet', origin='lower')
                
                # **NEW**: Conditionally plot the marker and legend
                if show_com:
                    ax2d.plot(center_x, center_y, 'r+', markersize=12, label=f'Center of Mass ({center_x:.1f}, {center_y:.1f})')
                    ax2d.legend()
                    
                plt.colorbar(im, ax=ax2d, label='Normalized Density')
                ax2d.set_title("2D Density Profile")
                st.pyplot(fig_2d)

                # --- Display 1D Profiles ---
                st.subheader("1D Density Profiles")
                prof_col1, prof_col2 = st.columns(2)
                with prof_col1:
                    st.write("Horizontal Profile")
                    st.line_chart(h_profile)
                with prof_col2:
                    st.write("Vertical Profile")
                    st.line_chart(v_profile)
                
                # --- Display 3D Plotly Surface Plot ---
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
                st.error("Cannot analyze an empty image. Please adjust the crop dimensions.")

    else:
        st.info("Awaiting image upload...")

if __name__ == "__main__":
    main()
