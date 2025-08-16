"""Geographic plotting utilities for navigation data"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import planar


class Geoplot:
    """Plot geographic data on interactive maps"""
    
    def __init__(self):
        self._colors = [
            '#1f77b4', 
            '#ff7f0e', 
            '#2ca02c', 
            '#d62728', 
            '#9467bd', 
            '#8c564b', 
            '#e377c2', 
            '#7f7f7f', 
            '#bcbd22', 
            '#17becf'
        ]
        self._font = {
            'family': 'Times New Roman',
            'size': 18
        }
        self._title = ''
        self._group_num = 0
        self._data = {}
    
    def set_title(self, title: str):
        """
        Set the figure title
        
        Parameters:
        -----------
        title : str
            Desired title
        """
        self._title = title
    
    def set_font(self, font: dict):
        """
        Set the figure font
        
        Parameters:
        -----------
        font : dict
            Desired font configuration
        """
        self._font = font
    
    def plot(self,
             lat: np.ndarray = None, 
             lon: np.ndarray = None, 
             alt: np.ndarray = None,
             time: np.ndarray = None,
             **kwargs):
        """
        Add plot to current figure window
        
        Parameters:
        -----------
        lat : np.ndarray
            Latitude [deg]
        lon : np.ndarray
            Longitude [deg]
        alt : np.ndarray, optional
            Altitude [m]
        time : np.ndarray, optional
            Time [s]
        **kwargs : dict
            color : str
                Desired plot color as hex code
            label : str
                Label for legend
            marker_size : float
                Desired relative marker size
        """
        if not isinstance(lat, np.ndarray):
            lat = np.array([lat], dtype=np.float64)
            lon = np.array([lon], dtype=np.float64)
            alt = np.array([alt], dtype=np.float64) if alt is not None else None
            time = np.array([time], dtype=np.float64) if time is not None else None
        
        # Check if lat and lon are input and same size
        if lat is None or lon is None:
            print('Must input both "lat" and "lon" as degrees!')
            print('Failed to add item!')
            return
        elif lat.size != lon.size:
            print('Size of "lat" and "lon" inputs must be equal!')
            print('Failed to add item!')
            return
        
        # Check if alt is input
        if alt is None or alt.size != lat.size:
            alt = np.zeros(lat.shape)
        
        # Check if time is input
        if time is None or time.size != lat.size:
            time = np.zeros(lat.shape)
        
        # Check keyword arguments
        if 'color' in kwargs:
            self._colors.insert(self._group_num, kwargs['color'])
        if 'label' in kwargs:
            label = kwargs['label']
        else:
            label = f'group{self._group_num}'
        if 'marker_size' in kwargs:
            marker_size = kwargs['marker_size']
        else:
            marker_size = 2.0
        
        # Add data
        self._data[f'lat{self._group_num}'] = lat
        self._data[f'lon{self._group_num}'] = lon
        self._data[f'alt{self._group_num}'] = alt
        self._data[f'time{self._group_num}'] = time
        self._data[f'label{self._group_num}'] = label
        self._data[f'marker_size{self._group_num}'] = marker_size
        self._group_num += 1
    
    def show(self):
        """Display the plot figure"""
        self._gen_dataframe()
        self._gen_figure()
        self._fig.show()
    
    def _gen_dataframe(self):
        """Generate pandas dataframe from the stored plot data dictionary"""
        # Combine data
        for i in range(self._group_num):
            if i == 0:
                LLAT = np.array([
                    self._data[f'lat{i}'],
                    self._data[f'lon{i}'],
                    self._data[f'alt{i}'],
                    self._data[f'time{i}'],
                    np.repeat([self._data[f'marker_size{i}']], self._data[f'lat{i}'].shape[0])
                ]).T
                self._sources = np.repeat([self._data[f'label{i}']], self._data[f'lat{i}'].shape[0])
            else:
                temp = np.array([
                    self._data[f'lat{i}'],
                    self._data[f'lon{i}'],
                    self._data[f'alt{i}'],
                    self._data[f'time{i}'],
                    np.repeat([self._data[f'marker_size{i}']], self._data[f'lat{i}'].shape[0])
                ]).T
                LLAT = np.vstack((LLAT, temp))
                self._sources = np.append(
                    self._sources, 
                    np.repeat([self._data[f'label{i}']], self._data[f'lat{i}'].shape[0])
                )
        
        # Generate dataframe
        self._df = pd.DataFrame(LLAT, columns=['lat', 'lon', 'alt', 'time', 'size'])
    
    def _gen_figure(self):
        """Create the geoplot figure and zoom to optimal settings"""
        self._fig = px.scatter_mapbox(
            self._df,
            lat="lat",
            lon="lon",
            color_discrete_sequence=self._colors,
            color=self._sources,
            hover_data=["alt", "time"],
            labels={
                "lat":  " Latitude [deg] ",
                "lon":  "Longitude [deg] ",
                "alt":  "   Altitude [m] ",
                "time": "       Time [s] ",
            },
            zoom=0.0,
            size='size',
            size_max=max(2 * self._df['size']),
        )
        
        all_pairs = []
        for lon, lat in zip(self._df.lon, self._df.lat):
            all_pairs.append((lon, lat))
        b_box = planar.BoundingBox(all_pairs)
        
        if b_box.is_empty:
            return 0, (0, 0)
        
        area = b_box.height * b_box.width
        zoom = np.interp(
            area,
            [0, 5**-10, 4**-10, 3**-10, 2**-10, 1**-10, 1**-5],
            [22, 19, 17.5, 16.5, 15, 12, 10]
        )
        center = b_box.center
        
        self._fig.update_layout(
            title=self._title,
            font=self._font,
            mapbox_style="white-bg",
            mapbox_layers=[
                {
                    "below": "traces",
                    "sourcetype": "raster",
                    "sourceattribution": "United States Geological Survey",
                    "source": [
                        "https://services.arcgisonline.com/arcgis/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
                    ],
                }
            ],
            mapbox=dict(center=go.layout.mapbox.Center(lat=center.y, lon=center.x), zoom=zoom),
            margin={"r": 0, "t": 0, "l": 0, "b": 0},
        )