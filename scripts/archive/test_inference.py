import joblib
import pandas as pd
from pathlib import Path
import os

ROOT = Path(r"d:\PythonProject1")
model_path = ROOT / "artifacts" / "best_model.pkl"
model = joblib.load(model_path)

payload = {
  'hotel': 'City Hotel',
  'lead_time': 30,
  'arrival_date_year': 2026,
  'arrival_date_month': 'July',
  'arrival_date_week_number': 28,
  'arrival_date_day_of_month': 10,
  'stays_in_weekend_nights': 0,
  'stays_in_week_nights': 2,
  'adults': 2,
  'children': 0,
  'babies': 0,
  'meal': 'BB',
  'country': 'PRT',
  'market_segment': 'Aviation',
  'distribution_channel': 'TA/TO',
  'is_repeated_guest': 1,
  'previous_cancellations': 3,
  'previous_bookings_not_canceled': 0,
  'reserved_room_type': 'A',
  'assigned_room_type': 'A',
  'booking_changes': 0,
  'deposit_type': 'No Deposit',
  'agent': None,
  'company': None,
  'days_in_waiting_list': 0,
  'customer_type': 'Group',
  'adr': 100,
  'required_car_parking_spaces': 1,
  'total_of_special_requests': 0
}

df = pd.DataFrame([payload])
print("DataFrame columns:", df.columns.tolist())

# Check raw predict proba
try:
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(df)[0]
        print(f"Raw Probabilities: {proba}")
    else:
        print("Model has no predict_proba")
except Exception as e:
    print("Error predicting:", e)
