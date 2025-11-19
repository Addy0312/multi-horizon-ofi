import pandas as pd
import numpy as np
import os
import glob
import argparse
from tqdm import tqdm

def process_day(msg_file, ob_file, output_dir, ticker, date):
    """
    Reads LOBSTER CSVs, cleans them, and saves as Parquet.
    """
    # 1. Read Message File
    # LOBSTER format: Time (sec from midnight), Event Type, Order ID, Size, Price, Direction
    # Note: 'Time' is strictly seconds after midnight.
    df_msg = pd.read_csv(msg_file, header=None, 
                         names=['time', 'event_type', 'order_id', 'size', 'price', 'direction'])
    
    # 2. Read Orderbook File
    # We assume level 10 based on your previous file
    level = 10
    cols = []
    for i in range(1, level+1):
        cols.extend([f'ask_price_{i}', f'ask_size_{i}', f'bid_price_{i}', f'bid_size_{i}'])
    
    df_ob = pd.read_csv(ob_file, header=None, names=cols)
    
    # 3. Merge
    if len(df_msg) != len(df_ob):
        print(f"Skipping {date}: Length mismatch ({len(df_msg)} vs {len(df_ob)})")
        return
    
    df = pd.concat([df_msg, df_ob], axis=1)
    
    # 4. Price Normalization (Standard LOBSTER scale is 10,000)
    # We convert to float for easier math later.
    price_cols = [c for c in df.columns if 'price' in c]
    df[price_cols] = df[price_cols] / 10000.0
    
    # 5. Time Filtering
    # 09:30:00 = 34200 seconds
    # 16:00:00 = 57600 seconds
    # Your filename suggests it is already chopped, but we enforce it to be safe.
    df = df[(df['time'] >= 34200) & (df['time'] <= 57600)].copy()
    
    # 6. Drop Crossed Markets (Safety check)
    # Even if 0% now, we drop any future bad rows
    df = df[df['bid_price_1'] < df['ask_price_1']]
    
    # 7. Add DateTime column (optional, but useful for visualization)
    # We create a dummy datetime based on the date string
    # This helps pandas plotting later
    base_date = pd.to_datetime(date)
    # Convert seconds to timedelta
    # We use 'unit=s' but LOBSTER time is float with microsecond precision
    df['datetime'] = base_date + pd.to_timedelta(df['time'], unit='s')
    
    # 8. Save
    # Create output folder: data/processed/AMZN/
    save_dir = os.path.join(output_dir, ticker)
    os.makedirs(save_dir, exist_ok=True)
    
    save_path = os.path.join(save_dir, f"{date}.parquet")
    df.to_parquet(save_path, compression='snappy', index=False)

def main():
    raw_dir = "data/raw"
    processed_dir = "data/processed"
    
    # Find message files
    msg_files = sorted(glob.glob(os.path.join(raw_dir, "*message*.csv")))
    
    print(f"Found {len(msg_files)} days to process.")
    
    for msg_path in tqdm(msg_files):
        # Parse filename to find matching orderbook
        # Format: Ticker_Date_Start_End_message_10.csv
        # Example: AMZN_2012-06-21_34200000_57600000_message_10.csv
        
        filename = os.path.basename(msg_path)
        parts = filename.split('_')
        ticker = parts[0]
        date = parts[1]
        
        # Construct Orderbook path
        ob_path = msg_path.replace("message", "orderbook")
        
        if not os.path.exists(ob_path):
            print(f"Warning: Orderbook file missing for {msg_path}")
            continue
            
        process_day(msg_path, ob_path, processed_dir, ticker, date)

if __name__ == "__main__":
    main()