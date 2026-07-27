#!/usr/bin/env bash
set -euo pipefail

BASE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXT="$BASE/data/raw/aviation/external"
mkdir -p "$EXT"

curl -L -o "$EXT/openflights_airports.csv" https://openflights.org/data/airports.dat

curl -L -o "$EXT/skytrax_airline_reviews.csv" https://raw.githubusercontent.com/quankiquanki/skytrax-reviews-dataset/master/data/airline.csv
curl -L -o "$EXT/skytrax_airport_reviews.csv" https://raw.githubusercontent.com/quankiquanki/skytrax-reviews-dataset/master/data/airport.csv
curl -L -o "$EXT/skytrax_lounge_reviews.csv" https://raw.githubusercontent.com/quankiquanki/skytrax-reviews-dataset/master/data/lounge.csv
curl -L -o "$EXT/skytrax_seat_reviews.csv" https://raw.githubusercontent.com/quankiquanki/skytrax-reviews-dataset/master/data/seat.csv

curl -L -o "$EXT/air_carrier_safety.xlsx" https://www.bts.gov/sites/bts.dot.gov/files/2025-02/table_02_09_022525.xlsx
curl -L -o "$EXT/commuter_airline_safety.xlsx" https://www.bts.gov/sites/bts.dot.gov/files/2025-02/table_02_10_022525.xlsx

curl -L -o "$EXT/ntsb_avall.zip" https://data.ntsb.gov/avdata/avall.zip

curl -L -o "$EXT/bts_db1b_fares.zip" https://www.bts.gov/sites/bts.dot.gov/files/docs/legacy/additional-attachment-files/DB1B.PUBLIC.201812.REL01.08APR2019.zip

# Mendeley dataset requires manual download due to access controls:
# https://data.mendeley.com/datasets/pc6fxc95h5/1
# Place these files into data/raw/aviation/external with exact filenames:
#   mendeley_annotated_abs_summ.csv
#   mendeley_annotated_sentiment.csv
#   mendeley_review_titles.csv
