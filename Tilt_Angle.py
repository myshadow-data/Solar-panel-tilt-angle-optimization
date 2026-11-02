import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Coordinates and parameters
latitude = 42.6965
longitude = 23.3220
panel_area = 1.6
nominal_efficiency = 0.20
ground_albedo = 0.20
NOCT = 45
temperature_coefficient = 0.004
U0 = 25
U1 = 6.84

# Tilt angle range
tilt_angles = np.arange(0, 90.5, 0.5)

# Data upload
file_path = "sofia_solar_data_2023.csv"
with open(file_path, "r", encoding="utf-8-sig") as file:
    lines = file.readlines()

header_row = None

for i, line in enumerate(lines):
    if "time" in line.lower() and "t2m" in line.lower():
        header_row = i
        break

if header_row is None:
    raise ValueError("Could not find the data header in the CSV file.")
data = pd.read_csv(
    file_path,
    skiprows=header_row,
    sep=None,
    engine="python"
)
data.columns = data.columns.str.strip()

# Convert numerical columns to numeric values
numeric_columns = [
    "P", "Gb(i)", "Gd(i)", "Gr(i)",
    "H_sun", "T2m", "WS10m"
]

for column in numeric_columns:
    data[column] = pd.to_numeric(data[column], errors="coerce")

print("Numerical columns are ready.")

data["time"] = pd.to_datetime(
    data["time"],
    format="%Y%m%d:%H%M"
)
time_intervals = data["time"].diff().dropna()
interval_hours = time_intervals.dt.total_seconds() / 3600
# Set the time interval in hours
dt_hours = 1.0

# Get the day number and UTC hour
day_number = data["time"].dt.dayofyear
utc_hour = (
    data["time"].dt.hour
    + data["time"].dt.minute / 60
)

# Calculate the fractional year angle
gamma = (2 * np.pi / 365) * (
    day_number - 1 + (utc_hour - 12) / 24
)

# Calculate solar declination in radians
declination = (
    0.006918
    - 0.399912 * np.cos(gamma)
    + 0.070257 * np.sin(gamma)
    - 0.006758 * np.cos(2 * gamma)
    + 0.000907 * np.sin(2 * gamma)
    - 0.002697 * np.cos(3 * gamma)
    + 0.001480 * np.sin(3 * gamma)
)

# Calculate the equation of time in minutes
equation_of_time = 229.18 * (
    0.000075
    + 0.001868 * np.cos(gamma)
    - 0.032077 * np.sin(gamma)
    - 0.014615 * np.cos(2 * gamma)
    - 0.040849 * np.sin(2 * gamma)
)

# Convert latitude and longitude to radians
latitude_rad = np.radians(latitude)
longitude_rad = longitude

# Calculate solar time
solar_time = utc_hour + (
    equation_of_time + 4 * longitude_rad
) / 60

# Calculate solar hour angle in radians
hour_angle = np.radians(15 * (solar_time - 12))

# Calculate the cosine of the zenith angle
cos_zenith = (
    np.sin(latitude_rad) * np.sin(declination)
    + np.cos(latitude_rad) * np.cos(declination)
    * np.cos(hour_angle)
)

# Keep cosine values between -1 and 1
cos_zenith = np.clip(cos_zenith, -1, 1)

# Calculate solar zenith and elevation in degrees
zenith_angle = np.degrees(np.arccos(cos_zenith))
solar_elevation = 90 - zenith_angle

# Calculate solar azimuth in degrees
azimuth_rad = np.arctan2(
    np.sin(hour_angle),
    np.cos(hour_angle) * np.sin(latitude_rad)
    - np.tan(declination) * np.cos(latitude_rad)
)

solar_azimuth = (np.degrees(azimuth_rad) + 180) % 360

# Add solar position values to the main dataset
data["solar_elevation"] = solar_elevation
data["solar_azimuth"] = solar_azimuth
data["solar_declination"] = np.degrees(declination)
data["hour_angle"] = np.degrees(hour_angle)

# Panel azimuth: south
panel_azimuth = 180

# Convert solar position data to NumPy arrays
solar_elevation_array = np.asarray(solar_elevation)
solar_azimuth_array = np.asarray(solar_azimuth)

# Calculate solar zenith angle
solar_zenith = 90 - solar_elevation_array

# Convert angles to radians
solar_zenith_rad = np.radians(solar_zenith)
solar_azimuth_rad = np.radians(solar_azimuth_array)
panel_azimuth_rad = np.radians(panel_azimuth)

tilt_rad = np.radians(tilt_angles)

# Calculate the cosine of the incidence angle
cos_incidence = (
    np.cos(solar_zenith_rad)[np.newaxis, :]
    * np.cos(tilt_rad)[:, np.newaxis]
    +
    np.sin(solar_zenith_rad)[np.newaxis, :]
    * np.sin(tilt_rad)[:, np.newaxis]
    * np.cos(
        solar_azimuth_rad - panel_azimuth_rad
    )[np.newaxis, :]
)

# Keep cosine values between -1 and 1
cos_incidence = np.clip(cos_incidence, -1, 1)

# Calculate the incidence angle in degrees
incidence_angle = np.degrees(
    np.arccos(cos_incidence)
)

# Find the index for a 32-degree tilt
tilt_32_index = np.where(tilt_angles == 32)[0][0]

# Get incidence angles for this tilt
incidence_32 = incidence_angle[tilt_32_index]
noon_index = np.argmax(solar_elevation)

# Get direct radiation on a horizontal surface
direct_horizontal = data["Gb(i)"].to_numpy()

# Create a mask for hours when the Sun is sufficiently high
valid_sun = cos_zenith > 0.065

# Calculate direct normal irradiance
dni = np.zeros(len(data))

dni[valid_sun] = (
    direct_horizontal[valid_sun]
    / cos_zenith[valid_sun]
)

# Calculate direct radiation on the tilted panel
direct_tilted = (
    np.maximum(cos_incidence, 0)
    * dni[np.newaxis, :]
)
tilt_32_index = np.where(tilt_angles == 32)[0][0]
direct_32 = direct_tilted[tilt_32_index]
max_direct = np.max(direct_32)

# Get diffuse radiation on a horizontal surface
diffuse_horizontal = data["Gd(i)"].to_numpy()

# Calculate the sky view factor for each tilt angle
sky_view_factor = (1 + np.cos(tilt_rad)) / 2

# Calculate diffuse radiation on tilted panels
diffuse_tilted = (
    sky_view_factor[:, np.newaxis]
    * diffuse_horizontal[np.newaxis, :]
)
diffuse_32 = diffuse_tilted[tilt_32_index]
horizontal_index = np.where(tilt_angles == 0)[0][0]
vertical_index = np.where(tilt_angles == 90)[0][0]

# Calculate global horizontal irradiance
ghi = direct_horizontal + diffuse_horizontal

# Calculate the ground view factor for each tilt angle
ground_view_factor = (1 - np.cos(tilt_rad)) / 2

# Calculate ground-reflected radiation on tilted panels
ground_reflected_tilted = (
    ground_albedo
    * ghi[np.newaxis, :]
    * ground_view_factor[:, np.newaxis]
)
ground_reflected_32 = ground_reflected_tilted[tilt_32_index]

# Calculate total solar radiation on tilted panels
total_tilted = (
    direct_tilted
    + diffuse_tilted
    + ground_reflected_tilted
)
total_32 = total_tilted[tilt_32_index]
max_total = np.max(total_32)

# Get ambient air temperature
air_temperature = data["T2m"].to_numpy()

# Calculate panel temperature using the NOCT model
panel_temperature_noct = (
    air_temperature[np.newaxis, :]
    + (total_tilted / 800) * (NOCT - 20)
)
temperature_noct_32 = panel_temperature_noct[tilt_32_index]
max_temperature_noct = np.max(temperature_noct_32)

# Get wind speed from the dataset
wind_speed = data["WS10m"].to_numpy()

# Prevent negative wind speed values
wind_speed = np.maximum(wind_speed, 0)

# Calculate panel temperature using the Faiman model
panel_temperature_faiman = (
    air_temperature[np.newaxis, :]
    + total_tilted
    / (
        U0
        + U1 * wind_speed[np.newaxis, :]
    )
)
temperature_faiman_32 = panel_temperature_faiman[tilt_32_index]
max_temperature_faiman = np.max(temperature_faiman_32)

# Calculate energy without temperature correction
energy_no_temperature = (
    total_tilted
    * panel_area
    * nominal_efficiency
    * dt_hours
)

# Calculate annual energy in kWh for each tilt angle
annual_energy_no_temperature = (
    np.sum(energy_no_temperature, axis=1) / 1000
)
# Calculate electrical efficiency using NOCT temperature
efficiency_noct = nominal_efficiency * (
    1 - temperature_coefficient
    * np.maximum(panel_temperature_noct - 25, 0)
)

# Keep efficiency within valid limits
efficiency_noct = np.clip(
    efficiency_noct,
    0,
    nominal_efficiency
)

# Calculate electrical energy using NOCT
energy_noct = (
    total_tilted
    * panel_area
    * efficiency_noct
    * dt_hours
)

# Calculate annual energy in kWh
annual_energy_noct = (
    np.sum(energy_noct, axis=1) / 1000
)

# Calculate electrical efficiency using Faiman temperature
efficiency_faiman = nominal_efficiency * (
    1 - temperature_coefficient
    * np.maximum(panel_temperature_faiman - 25, 0)
)

# Keep efficiency within valid limits
efficiency_faiman = np.clip(
    efficiency_faiman,
    0,
    nominal_efficiency
)

# Calculate electrical energy using Faiman
energy_faiman = (
    total_tilted
    * panel_area
    * efficiency_faiman
    * dt_hours
)

# Calculate annual energy in kWh
annual_energy_faiman = (
    np.sum(energy_faiman, axis=1) / 1000
)

print(
    "Annual energy without temperature correction:",
    round(annual_energy_no_temperature[tilt_32_index], 2),
    "kWh"
)

print(
    "Annual energy using NOCT:",
    round(annual_energy_noct[tilt_32_index], 2),
    "kWh"
)

print(
    "Annual energy using Faiman:",
    round(annual_energy_faiman[tilt_32_index], 2),
    "kWh"
)

print(
    "Nominal efficiency:",
    round(nominal_efficiency * 100, 2),
    "%"
)

print(
    "Average NOCT efficiency:",
    round(np.mean(efficiency_noct[tilt_32_index]) * 100, 2),
    "%"
)

print(
    "Average Faiman efficiency:",
    round(np.mean(efficiency_faiman[tilt_32_index]) * 100, 2),
    "%"
)

# Find the index of maximum annual energy for each scenario
best_index_no_temperature = np.argmax(
    annual_energy_no_temperature
)

best_index_noct = np.argmax(
    annual_energy_noct
)

best_index_faiman = np.argmax(
    annual_energy_faiman
)

# Get the optimal tilt angles
best_tilt_no_temperature = tilt_angles[
    best_index_no_temperature
]

best_tilt_noct = tilt_angles[
    best_index_noct
]

best_tilt_faiman = tilt_angles[
    best_index_faiman
]

# Get the maximum annual energy values
best_energy_no_temperature = annual_energy_no_temperature[
    best_index_no_temperature
]

best_energy_noct = annual_energy_noct[
    best_index_noct
]

best_energy_faiman = annual_energy_faiman[
    best_index_faiman
]

print("Optimal tilt angle without temperature correction:",
      best_tilt_no_temperature, "degrees")

print("Optimal tilt angle using NOCT:",
      best_tilt_noct, "degrees")

print("Optimal tilt angle using Faiman:",
      best_tilt_faiman, "degrees")

print()

print("Maximum annual energy without temperature correction:",
      round(best_energy_no_temperature, 2), "kWh")

print("Maximum annual energy using NOCT:",
      round(best_energy_noct, 2), "kWh")

print("Maximum annual energy using Faiman:",
      round(best_energy_faiman, 2), "kWh")

optimization_results = pd.DataFrame({
    "Scenario": [
        "Without temperature correction",
        "NOCT",
        "Faiman"
    ],
    "Optimal tilt angle (degrees)": [
        best_tilt_no_temperature,
        best_tilt_noct,
        best_tilt_faiman
    ],
    "Annual energy (kWh)": [
        best_energy_no_temperature,
        best_energy_noct,
        best_energy_faiman
    ]
})

display(optimization_results)

plt.figure(figsize=(12, 6))

plt.plot(
    tilt_angles,
    annual_energy_no_temperature,
    label="Without temperature correction"
)

plt.plot(
    tilt_angles,
    annual_energy_noct,
    label="NOCT"
)

plt.plot(
    tilt_angles,
    annual_energy_faiman,
    label="Faiman"
)

plt.xlabel("Panel tilt angle (degrees)")
plt.ylabel("Annual energy production (kWh)")
plt.title("Annual Energy Production vs Panel Tilt Angle")

plt.legend()
plt.grid(True)

plt.show()


scenarios = [
    "Without temperature correction",
    "NOCT",
    "Faiman"
]

energy_values = [
    best_energy_no_temperature,
    best_energy_noct,
    best_energy_faiman
]


plt.figure(figsize=(10, 6))

bars = plt.bar(
    scenarios,
    energy_values
)
for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        height + 3,
        f"{height:.2f} kWh",
        ha="center"
    )
plt.title("Annual Energy Production Comparison")
plt.xlabel("Scenario")
plt.ylabel("Annual Energy Production (kWh)")
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.ylim(0, max(energy_values) * 1.15)
plt.xticks(rotation=10)
plt.tight_layout()
plt.show()