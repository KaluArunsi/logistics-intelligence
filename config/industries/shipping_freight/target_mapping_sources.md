# Shipping_Freight Target Mapping Sources

This reference captures external evidence used to map business-value targets to workers.

## Primary Sources
- World Bank LPI (customs, infrastructure, tracking/tracing, timeliness): https://lpi.worldbank.org/international
- WCO Time Release Study Guide v4 (clearance time, border process performance): https://www.wcoomd.org/es-es/topics/facilitation/instrument-and-tools/tools/time-release-study.aspx
- UNCTAD vessel turnaround KPI (waiting + berth/service time): https://sft-framework.unctad.org/key-performance-indicator/maritime-vessel-turnaround-time
- UNCTAD port efficiency/turnaround discussion: https://unctad.org/news/container-ports-fastest-busiest-and-best-connected
- BTS Freight Mobility Initiative (county-to-county travel-time percentiles): https://www.bts.gov/newsroom/bts-unveils-new-statistical-program-trucking-and-freight-mobility
- BTS Freight TSI (mode-level freight output signal): https://www.bts.gov/newsroom/december-2025-freight-transportation-services-index-tsi-fell-06-november-2025-and-rose-04
- FMCSA SMS (BASICs and intervention thresholds linked to crash risk): https://ai.fmcsa.dot.gov/SMS/HelpCenter/Index.aspx
- WHO cold-chain temperature mapping and controls: https://www.who.int/publications/m/item/cold-chain-equipment-and-dry-store-temperature-mapping-tool
- WHO temperature-controlled vaccine distribution constraints: https://www.who.int/teams/health-product-policy-and-standards/standards-and-specifications/norms-and-standards/vaccine-standardization/extended-controlled-temperature-conditions

## Category-to-Target Intent
- `customs_compliance`: clearance delay, inspection/hold risk, compliance risk tier.
- `cross_border_flow`: border throughput, queue/wait time, crossing delay.
- `trade_volume_mix`: trade value/tonnage/TEU forecast by commodity/mode.
- `port_terminal_congestion`: berth wait, terminal dwell, congestion bands.
- `ocean_schedule_reliability`: schedule adherence, arrival delay, timeliness risk.
- `inland_waterway_flow`: lock/route delay, draft constraints, flow volume.
- `rail_intermodal_flow`: terminal dwell, transfer time, intermodal throughput.
- `trucking_capacity`: county-pair travel time, lane capacity tightness, availability.
- `fleet_utilization`: utilization ratio, idle/empty movement risk, asset productivity.
- `eta_delay_risk`: late-arrival risk flag, ETA error band.
- `last_mile_sla`: SLA breach risk, exception likelihood.
- `lane_cost_yield`: lane cost/yield/margin forecasting.
- `claims_damage_risk`: damage/claim likelihood and severity bands.
- `carrier_safety_risk`: incident/crash/violation risk from safety signals.
- `cold_chain_integrity`: temperature excursion and integrity-break risk.
