import os
import pandas as pd
from sqlalchemy import create_engine, text
from src.Bonus_Allocation_System.logging import logger
from src.Bonus_Allocation_System.entity.config_entity import DataIngestionConfig
import zipfile
from pathlib import Path


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def download_file(self):
        if not os.path.exists(self.config.local_data_file):
            
            engine = create_engine("mysql+pymysql://{user}:{pw}@localhost/{db}"
                       .format(user = "root",    # user
                               pw = "nitd123",      # passwrd
                               db = "project")) 
            
            sql = 'SELECT * FROM project_data'
            
            my_project = pd.read_sql_query(text(sql), engine.connect()).convert_dtypes()
            project = "project_data.csv"
            
           
            my_project.to_csv(self.config.local_data_file,index=False)
            logger.info(f"Required file is downloaded")
        else:
            logger.info(f"file already exists ")
            
    def extract_zip_file(self):
    
        unzip_path = self.config.unzip_dir
        os.makedirs(unzip_path, exist_ok=True)

        with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
            zip_ref.extractall(unzip_path)

        # Assuming there's only one CSV file inside the ZIP
        extracted_file = os.path.join(unzip_path, next(iter(zip_ref.namelist())))

        # Read the extracted CSV file
        try:
            df = pd.read_csv(extracted_file)
        except pd.errors.ParserError as e:
            raise ValueError(f"Error parsing extracted CSV file: {e}") from e

        # Optionally specify a new CSV filename (if needed)
        output_csv = os.path.join(self.config.unzip_dir, 'extracted_data.csv')  # Adjust path if needed
        df.to_csv(output_csv, index=False)

        logger.info(f"Extracted and converted data to CSV: {output_csv}")

            
