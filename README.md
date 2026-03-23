# afd-analogs

Historical weather pattern matching through NLP analysis of NWS Area Forecast Discussions.

Takes a current AFD and finds the most semantically similar historical discussions from an archive, then shows what weather actually occurred on those dates.

## Setup

```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

## Data Sources

- **AFDs**: [Iowa Environmental Mesonet](https://mesonet.agron.iastate.edu/wx/afos/list.phtml) NWS text archives
- **Observations**: [NOAA Local Climatological Data](https://www.ncei.noaa.gov/cdo-web/) (LCD) via NCEI API

Target WFO: BGM (Binghamton, NY)

## Team

- Leif Rogers (lfr42)
- Jingyu Xu (jx62)
- Eric Gao (xg255)
