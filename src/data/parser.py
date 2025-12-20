"""
NOAA ISD Data Parser
Parses NOAA Integrated Surface Database fixed-width format files
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import yaml
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


class NOAAParser:
    """Parser for NOAA ISD fixed-width format data files"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize parser with configuration"""
        self.config_path = Path(config_path)
        self.load_config()
        
    def load_config(self):
        """Load configuration from YAML file"""
        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.raw_path = Path(self.config['data']['raw_path'])
        self.stations = self.config['stations']
        self.field_positions = self.config['noaa_isd_format']
        self.missing_codes = self.config['processing']['missing_codes']
        
    def parse_line(self, line: str) -> Dict:
        """Parse a single line of NOAA ISD format
        
        Args:
            line: Raw line from NOAA ISD file
            
        Returns:
            Dictionary with parsed values
        """
        try:
            # Extract fixed-width fields
            station_id = line[4:15].strip()
            date_str = line[15:23]
            time_str = line[23:27]
            
            # Parse datetime
            year = int(date_str[0:4])
            month = int(date_str[4:6])
            day = int(date_str[6:8])
            hour = int(time_str[0:2])
            minute = int(time_str[2:4])
            
            datetime_str = f"{year}-{month:02d}-{day:02d} {hour:02d}:{minute:02d}"
            
            # Parse meteorological variables
            temp_raw = line[87:92].strip()
            dewp_raw = line[93:98].strip()
            
            # Temperature (stored as temp*10 in Celsius)
            try:
                temperature = float(temp_raw) / 10.0 if temp_raw and temp_raw != '+9999' else np.nan
            except:
                temperature = np.nan
            
            # Dew point (stored as dewp*10 in Celsius)
            try:
                dewpoint = float(dewp_raw) / 10.0 if dewp_raw and dewp_raw != '+9999' else np.nan
            except:
                dewpoint = np.nan
            
            # Extract additional fields if available
            wind_dir_raw = line[60:63].strip() if len(line) > 63 else ''
            wind_speed_raw = line[65:69].strip() if len(line) > 69 else ''
            
            try:
                wind_direction = float(wind_dir_raw) if wind_dir_raw and wind_dir_raw != '999' else np.nan
            except:
                wind_direction = np.nan
            
            try:
                # Wind speed stored as m/s * 10
                wind_speed = float(wind_speed_raw) / 10.0 if wind_speed_raw and wind_speed_raw != '9999' else np.nan
            except:
                wind_speed = np.nan
            
            # Sea level pressure (if available)
            if len(line) > 104:
                slp_raw = line[99:104].strip()
                try:
                    sea_level_pressure = float(slp_raw) / 10.0 if slp_raw and slp_raw != '99999' else np.nan
                except:
                    sea_level_pressure = np.nan
            else:
                sea_level_pressure = np.nan
            
            return {
                'station_id': station_id,
                'datetime': datetime_str,
                'year': year,
                'month': month,
                'day': day,
                'hour': hour,
                'minute': minute,
                'temperature': temperature,
                'dewpoint': dewpoint,
                'wind_direction': wind_direction,
                'wind_speed': wind_speed,
                'sea_level_pressure': sea_level_pressure
            }
        except Exception as e:
            return None
    
    def parse_file(self, file_path: Path) -> pd.DataFrame:
        """Parse a single NOAA ISD file
        
        Args:
            file_path: Path to NOAA ISD file
            
        Returns:
            DataFrame with parsed data
        """
        records = []
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    if len(line) > 100:  # Valid NOAA ISD lines are > 100 chars
                        parsed = self.parse_line(line)
                        if parsed:
                            records.append(parsed)
        except Exception as e:
            print(f"Error parsing {file_path}: {e}")
            return pd.DataFrame()
        
        if not records:
            return pd.DataFrame()
        
        df = pd.DataFrame(records)
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.set_index('datetime').sort_index()
        
        return df
    
    def parse_station_year(self, station_id: str, year: int) -> pd.DataFrame:
        """Parse data for specific station and year
        
        Args:
            station_id: Station ID (e.g., '722020-12839')
            year: Year to parse
            
        Returns:
            DataFrame with parsed data
        """
        # Find file for this station/year
        year_path = self.raw_path / str(year)
        
        # Try different possible file patterns
        possible_patterns = [
            year_path / f"{station_id}-{year}" / f"{station_id}-{year}",  # In subfolder with same name
            year_path / f"{station_id}-{year}",  # Direct file
            year_path / station_id / f"{station_id}-{year}",  # In subfolder
        ]
        
        for file_path in possible_patterns:
            if file_path.exists() and file_path.is_file():
                return self.parse_file(file_path)
        
        # Try recursive search as last resort
        files = list(year_path.rglob(f"*{station_id}*{year}*"))
        files = [f for f in files if f.is_file()]
        if files:
            return self.parse_file(files[0])
        
        print(f"Warning: File not found for {station_id} {year}")
        return pd.DataFrame()
    
    def parse_all_years(self, station_id: Optional[str] = None) -> pd.DataFrame:
        """Parse all years for one or all stations
        
        Args:
            station_id: Station ID to parse (None = all stations)
            
        Returns:
            Combined DataFrame with all data
        """
        all_data = []
        
        start_year = self.config['time_range']['start_year']
        end_year = self.config['time_range']['end_year']
        
        # Get station list
        if station_id:
            stations = [station_id]
        else:
            stations = list(self.stations.keys())
        
        total_files = len(stations) * (end_year - start_year + 1)
        
        with tqdm(total=total_files, desc="Parsing files") as pbar:
            for station in stations:
                station_name = self.stations[station]['name']
                
                for year in range(start_year, end_year + 1):
                    pbar.set_description(f"Parsing {station_name} {year}")
                    
                    df = self.parse_station_year(station, year)
                    
                    if not df.empty:
                        df['station_code'] = station
                        df['station_name'] = station_name
                        df['city'] = self.stations[station]['city']
                        df['climate_zone'] = self.stations[station]['climate_zone']
                        all_data.append(df)
                    
                    pbar.update(1)
        
        if not all_data:
            print("No data parsed!")
            return pd.DataFrame()
        
        # Combine all dataframes
        combined_df = pd.concat(all_data, axis=0).sort_index()
        
        return combined_df
    
    def calculate_derived_variables(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate derived meteorological variables
        
        Args:
            df: DataFrame with basic variables
            
        Returns:
            DataFrame with additional derived variables
        """
        # Relative Humidity (approximation using Magnus formula)
        def calc_relative_humidity(temp, dewpoint):
            """Calculate relative humidity from temperature and dewpoint"""
            if pd.isna(temp) or pd.isna(dewpoint):
                return np.nan
            
            # Magnus formula constants
            a = 17.27
            b = 237.7
            
            alpha_temp = (a * temp) / (b + temp)
            alpha_dewp = (a * dewpoint) / (b + dewpoint)
            
            rh = 100 * (np.exp(alpha_dewp - alpha_temp))
            return np.clip(rh, 0, 100)
        
        df['relative_humidity'] = df.apply(
            lambda row: calc_relative_humidity(row['temperature'], row['dewpoint']),
            axis=1
        )
        
        # Heat Index (when temp > 26.7°C)
        def calc_heat_index(temp, rh):
            """Calculate heat index (feels-like temperature in hot conditions)"""
            if pd.isna(temp) or pd.isna(rh) or temp < 26.7:
                return temp
            
            # Convert to Fahrenheit for formula
            T = temp * 9/5 + 32
            R = rh
            
            HI = -42.379 + 2.04901523*T + 10.14333127*R - 0.22475541*T*R
            HI += -0.00683783*T*T - 0.05481717*R*R + 0.00122874*T*T*R
            HI += 0.00085282*T*R*R - 0.00000199*T*T*R*R
            
            # Convert back to Celsius
            return (HI - 32) * 5/9
        
        df['heat_index'] = df.apply(
            lambda row: calc_heat_index(row['temperature'], row['relative_humidity']),
            axis=1
        )
        
        return df
    
    def save_processed(self, df: pd.DataFrame, filename: str = "processed_data.parquet"):
        """Save processed data to file
        
        Args:
            df: Processed DataFrame
            filename: Output filename
        """
        output_path = Path(self.config['data']['processed_path']) / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save as Parquet (efficient format)
        df.to_parquet(output_path, index=True, compression='snappy')
        print(f"Saved to {output_path}")
        print(f"Shape: {df.shape}")
        print(f"Size: {output_path.stat().st_size / (1024**2):.2f} MB")


if __name__ == "__main__":
    # Example usage
    parser = NOAAParser()
    
    print("Parsing all NOAA ISD data...")
    df = parser.parse_all_years()
    
    if not df.empty:
        print("\n=== Parsing Complete ===")
        print(f"Total records: {len(df):,}")
        print(f"Date range: {df.index.min()} to {df.index.max()}")
        print(f"Stations: {df['station_name'].unique()}")
        print(f"\nColumns: {list(df.columns)}")
        print(f"\nSample data:\n{df.head()}")
        
        # Calculate derived variables
        print("\nCalculating derived variables...")
        df = parser.calculate_derived_variables(df)
        
        # Save processed data
        parser.save_processed(df)
