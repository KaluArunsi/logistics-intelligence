# Trucking_Delivery Target Mapping Sources

This reference captures external evidence used to map business-value targets to trucking and last-mile workers.

## Primary Sources
- USDOT Open Data Hub (road, trucking, mobility, safety): https://data.transportation.gov/
- BTS Border Crossing Entry Data (land-border crossing times/volumes): https://www.bts.gov/browse-statistical-products-and-data/border-crossing-data/border-crossingentry-data
- BTS Commodity Flow Survey (CFS): https://www.bts.gov/cfs
- BTS Freight Analysis Framework (FAF): https://www.bts.gov/faf
- FMCSA Data Dissemination Program (motor-carrier safety/compliance): https://www.fmcsa.dot.gov/registration/fmcsa-data-dissemination-program
- FMCSA SMS/CSA downloads (inspection/crash/safety): https://ai.fmcsa.dot.gov/SMS/Tools/Downloads.aspx
- NHTSA FARS (road safety incidents): https://www.nhtsa.gov/research-data/fatality-analysis-reporting-system-fars
- Eurostat Road Freight / Road Transport indicators: https://ec.europa.eu/eurostat/data/database
- UK Department for Transport road freight statistics: https://www.gov.uk/government/statistics/road-freight-statistics-2024
- Hong Kong open transport/courier datasets: https://data.gov.hk/en-data/
- Singapore transport public datasets: https://data.gov.sg/
- World Bank Logistics Performance Index: https://lpi.worldbank.org/international
- OECD / ITF transport statistics: https://www.itf-oecd.org/data-statistics

## Enterprise KPI Benchmark References (non-government)
- DHL investor and annual reports: https://www.dhl.com/global-en/home/about-us/investor-relations.html
- FedEx investor reports: https://investors.fedex.com/
- Amazon Last Mile Routing Research Challenge: https://github.com/amzn/amazon-last-mile-challenges

## Category-to-Target Intent
- `route_eta_reliability`: late-arrival flag, ETA error bands, on-time probability.
- `dispatch_capacity_balance`: dispatch backlog, unassigned load pressure, capacity index.
- `last_mile_sla`: SLA breach risk, failed-attempt risk, delivery exception likelihood.
- `driver_safety_compliance`: violation risk, incident risk, inspection failure probability.
- `fleet_utilization`: utilization %, idle exposure, empty-mile ratio.
- `vehicle_maintenance_risk`: maintenance due risk, breakdown risk, failure probability.
- `fuel_energy_efficiency`: fuel burn rate, MPG/energy intensity per trip.
- `lane_cost_yield`: cost-per-mile, lane margin and yield.
- `pickup_dropoff_reliability`: stop punctuality and missed-slot risk.
- `parcel_exception_risk`: damage/claim/exception probability.
- `reverse_logistics_returns`: return pickup and reverse-cycle duration.
- `cross_border_trucking`: crossing time and queue-delay forecasting.
- `urban_traffic_risk`: congestion and route-disruption risk.
- `cold_chain_last_mile`: temperature excursion and integrity breach risk.
- `workforce_shift_planning`: staffing gap and overtime pressure forecasts.
