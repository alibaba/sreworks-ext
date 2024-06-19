
#import glob
import polars as pl
#import os
from collections import Counter
import json

# Base class
class BaseLoader:
    #This csv separator should never be found. 
    #We try to disable polars from doing csv splitting.
    #Instead we do it manually to get it correctly done. 
    _csv_separator = "\a" 
    _mandatory_columns = ["m_message", "m_timestamp"]
    
    def __init__(self, filename, df=None, df_seq=None):
        self.filename = filename
        self.df = df #Event level dataframe
        self.df_seq = df_seq #sequence level dataframe

    def load(self):
        print(f"WARNING! You are using dummy loader. This results in dataframe with single column only titled: m_message"
              f"Consider implmenting dataset specific loader")
        self.df = pl.read_csv(self.filename, has_header=False, infer_schema_length=0, 
                        separator=self._csv_separator, ignore_errors=True)
        self.df = self.df.rename({"column_1": "m_message"})
        
    def preprocess(self):
        raise NotImplementedError

    def execute(self):
        if self.df is None:
            self.load()
        self.preprocess()
        self.check_for_nulls() 
        self.check_mandatory_columns()
        self.add_ano_col()
        return self.df
    
    def add_ano_col(self):
        # Check if the 'normal' column exists
        if self.df is not None and  "normal" in self.df.columns:
            # Create the 'anomaly' column by inverting the boolean values of the 'normal' column
            self.df = self.df.with_columns(pl.col("normal").not_().alias("anomaly"))
        if self.df_seq is not None and "normal" in self.df_seq:
            # Create the 'anomaly' column by inverting the boolean values of the 'normal' column
            self.df_seq = self.df_seq.with_columns(pl.col("normal").not_().alias("anomaly"))

        # Check if the 'anomaly' column exists but no normal column
        if self.df is not None and  "anomaly" in self.df.columns and not "normal" in self.df.columns:
            # Create the 'normal' column by inverting the boolean values of the 'anomaly' column
            self.df = self.df.with_columns(pl.col("anomaly").not_().alias("normal"))
        #self._mandatory_columns = ["m_message"]

    
    def check_for_nulls(self):
        null_counts = {}  # Dictionary to store count of nulls for each column
        for col in self.df.columns:
            null_count = self.df.filter(self.df[col].is_null()).shape[0]
            if null_count > 0:
                null_counts[col] = null_count
        # Print the results
        if null_counts:
            for col, count in null_counts.items():
                print(f"WARNING! Column '{col}' has {count} null values out of {len(self.df)}. You have 4 options:"
                        f" 1) Do nothing and hope for the best"
                        f", 2) Drop the column that has nulls"
                        f", 3) Filter out rows that have nulls"
                        f", 4) Investigate and fix your Loader")
                print(f"To investigate: <DF_NAME>.filter(<DF_NAME>['{col}'].is_null())")

    def check_mandatory_columns(self):
        missing_columns = [col for col in self._mandatory_columns if col not in self.df.columns]
        if missing_columns:
            raise ValueError(f"Missing mandatory columns: {', '.join(missing_columns)}")
                  
        if 'm_time_stamp' in self._mandatory_columns and not isinstance(self.df.column("m_time_stamp").dtype, pl.datatypes.Datetime):
            raise TypeError("Column 'm_time_stamp' is not of type Polars.Datetime")

    def _split_and_unnest(self, field_names):
        #split_cols = self.df["column_1"].str.splitn(" ", n=len(field_names))
        split_cols = self.df.select(pl.col("column_1")).to_series().str.splitn(" ", n=len(field_names))
        split_cols = split_cols.struct.rename_fields(field_names)
        split_cols = split_cols.alias("fields")
        split_cols = split_cols.to_frame()
        self.df = split_cols.unnest("fields")
      
    def lines_not_starting_with_pattern(self, pattern=None):
        if self.df is None:
            self.load()
        if pattern is None:
            pattern = self.event_pattern

        # Filter lines that do not start with the pattern.
        non_matching_lines_df = self.df.filter(~pl.col("column_1").str.contains(pattern))
        # Filter lines that do start with the pattern.
        matching_lines_df = self.df.filter(pl.col("column_1").str.contains(pattern))
        
        # Get the number of lines that do not start with the pattern.
        non_matching_lines_count = non_matching_lines_df.shape[0]
        # Get the number of lines that do start with the pattern.
        matching_lines_count = matching_lines_df.shape[0]
            
        return non_matching_lines_df, non_matching_lines_count, matching_lines_df, matching_lines_count 

    def reduce_dataframes(self, frac=0.5, random_state=42):
        # If df_sequences is present, reduce its size
        if hasattr(self, 'df_seq') and self.df_seq is not None:
             # Sample df_seq
            df_seq_temp = self.df_seq.sample(fraction=frac, seed=random_state)

            # Check if df_seq still has at least one row
            if len(df_seq_temp) == 0:
                # If df_seq is empty after sampling, randomly select one row from the original df_seq
                self.df_seq = self.df_seq.sample(n=1)
            else:
                self.df_seq = df_seq_temp
            # Update df to include only the rows that have seq_id values present in the filtered df_seq
            self.df = self.df.filter(pl.col("seq_id").is_in(self.df_seq["seq_id"]))

            #self.df_seq = self.df_seq.sample(fraction=frac)
            # Update df to include only the rows that have seq_id values present in the filtered df_sequences
            #self.df = self.df.filter(pl.col("seq_id").is_in(self.df_seq["seq_id"]))
        else:
            # If df_sequences is not present, just reduce df
            self.df = self.df.sample(fraction=frac, seed=random_state)

        return self.df
    
    def parse_json(self, json_line):
        json_data = json.loads(json_line)
        return pl.DataFrame([json_data])

#Process log files created with the GELF logging driver.
# 
class GELFLoader(BaseLoader):
    def load(self):
        with open(self.filename, 'r') as file:
            lines = file.readlines()

        # Assuming each line in the file is a separate JSON object
        json_frames = [self.parse_json(line) for line in lines]
        self.df = pl.concat(json_frames)

    def preprocess(self):
        # Rename some columns to match the expected column names and parse datetime
        self.df = self.df.with_columns(
            pl.col("message").alias("m_message")
            ).drop("message")
        
        parsed_timestamps = self.df.select(
            pl.col("@timestamp").str.strptime(pl.Datetime, strict=False).alias("m_timestamp")
        ).drop("@timestamp")
        self.df = self.df.with_columns(parsed_timestamps)