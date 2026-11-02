# Mobile Network Planning & RF Optimization

This repository contains a Streamlit-based simulation for planning and optimizing a cellular network in a crowded railway station.

The project models:

- 1,000–10,000 active passengers
- Macro and micro/pico cell placement
- Passenger density around platforms and concourses
- Path loss and received signal power
- Frequency reuse factors `K = 1`, `3`, and `7`
- Co-channel interference
- SINR, capacity, throughput, and congestion
- Frequency reallocation and dynamic power optimization

## Project files

The application is located in [`Downloads/wirless project/`](./Downloads/wirless%20project/).

- [`app.py`](./Downloads/wirless%20project/app.py) — Streamlit dashboard
- [`rf_engine.py`](./Downloads/wirless%20project/rf_engine.py) — RF, SINR, interference, and capacity calculations
- [`traffic_sim.py`](./Downloads/wirless%20project/traffic_sim.py) — Passenger traffic and cell placement
- [`optimizer.py`](./Downloads/wirless%20project/optimizer.py) — Network optimization
- [`requirements.txt`](./Downloads/wirless%20project/requirements.txt) — Python dependencies

## Run the application

From PowerShell:

```powershell
Set-Location "Downloads\wirless project"

& "C:\Program Files\Python314\python.exe" -m pip install -r requirements.txt
& "C:\Program Files\Python314\python.exe" -m streamlit run app.py
```

Open `http://localhost:8501` in your browser.

## Project purpose

The simulation demonstrates how cell capacity, frequency reuse, co-channel interference, SINR, and congestion are related in a dense railway-station environment. It is a planning-level model for FRMCS, 4G, and 5G fundamentals, not a complete commercial protocol-stack simulator.
