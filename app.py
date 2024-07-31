from __future__ import division
import streamlit as st 
import matplotlib.pyplot as plt

from data_handling import Data_Handling
from graph_drawing import Graph_Drawing

data_handling = Data_Handling()
graph_drawing = Graph_Drawing()

st.title('Customer Segmenting App')

file = st.file_uploader('Upload your file:', ['csv', 'xlsx'])
if 'stage' not in st.session_state:
    st.session_state.stage = 0

def click_button(stage):
    st.session_state.stage = stage

if file:
    raw_data = data_handling.get_raw(file)
    if not raw_data.empty:
        st.dataframe(raw_data)
        try:
            df = data_handling.create_dataframe(raw_data)
            st.success('Dataframe created successfully.')
        except KeyError as ke:
            st.error(f'You need columns with such names: AccountID, CloseDate, DealValue, DealStage')
            st.stop()
        except Exception as e:
            st.error(f'Error creating dataframe: {type(e)}')
            st.stop()
            
        if st.button('Run RFM Segmentation'):
            click_button(1)
        
        if st.session_state.stage >= 1:
            # Creates RFM dataframe for the segmentation
            rfm_data = data_handling.create_rfm_dataframe(df)

            # Creates dataframe with clusters from kmeans
            kmeans_data, cluster_centers, silhouette_score = data_handling.create_kmeans_dataframe(rfm_data)
            st.header('Silhouette Score: {:0.2f}'.format(silhouette_score))

            # Creates graphs 
            for component, color in zip(['Recency', 'Frequency', 'Monetary'], ['blue', 'green', 'orange']):
                figure = graph_drawing.rfm_component_graph(rfm_data, component, color)
                st.pyplot(figure)
                plt.close()
                
            if st.button('Show treemap'):
                click_button(2)
            
            if st.session_state.stage >= 2:
                # Creates treemaps
                tree_figure = graph_drawing.treemap_drawing(cluster_centers)
                st.pyplot(tree_figure)
            
            if st.button('Show scatterplot'):
                click_button(3)
            
            if st.session_state.stage >= 3:
                scatter_figure = graph_drawing.scatter_3d_drawing(kmeans_data)
                st.plotly_chart(scatter_figure)
