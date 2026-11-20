# Architecture

Provider observations enter the price engine. Calendar-aligned completed candles supply causal features and target timestamps. The live worker records forecasts in SQLite by origin, horizon and model. Settlement joins the exact target observation, and model evaluation compares paired forecast errors. FastAPI exposes health and presentation endpoints. Production platform scheduling is executed by systemd on the server. The separate public demo serves a historical snapshot.
