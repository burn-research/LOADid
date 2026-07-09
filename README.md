# LOADid
Framework for identifying equipments behind load profiles.

## Data

Consumption data is sourced from the [Fluvius open data portal](https://opendata.fluvius.be/pages/homepage_v30/). The datasets extracted are:

- [Hourly gas meters](https://opendata.fluvius.be/explore/assets/1_50-verbruiksprofielen-dm-gas-uurwaarden-voor-een-volledig-jaar/)
- [Quarter-hourly electricity meters](https://opendata.fluvius.be/explore/assets/1_50-verbruiksprofielen-dm-elek-kwartierwaarden-voor-een-volledig-jaar/)

Those have been unzipped in [Elec](Data/Raw/Elec/) and [Gas](Data/Raw/Gas/) folders and preprocessed in the [Preprocess](Preprocess/) folder.

Weather data is sourced from the data portal of the [Royal Meteorological Institute of Belgium](https://opendata.meteo.be/). 

