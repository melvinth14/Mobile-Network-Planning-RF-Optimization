# Mobile Network Planning & RF Optimization for a Crowded Railway Station

A Streamlit-based planning simulation for dense railway-station cellular networks. It models passenger clustering, macro/micro-cell placement, path loss, received power, frequency reuse, co-channel interference, SINR, capacity, congestion, and heuristic RF optimization.

## Features

- Simulates 1,000–10,000 active passengers.
- Models platform/train clusters and concourse traffic.
- Supports macro and micro/pico cells.
- Calculates FSPL/log-distance propagation, received power, CCI, SINR, throughput, and utilization.
- Supports frequency reuse factors `K = 1`, `3`, and `7`.
- Applies frequency reallocation and dynamic power control.
- Provides coverage, SINR, topology, and per-cell dashboards.

## Run locally

```powershell
& "C:\Program Files\Python314\python.exe" -m pip install -r requirements.txt
& "C:\Program Files\Python314\python.exe" -m streamlit run app.py
```

Open `http://localhost:8501` in a browser.

This is a planning-level simulation for FRMCS/4G/5G fundamentals, not a complete 3GPP protocol-stack simulator.
