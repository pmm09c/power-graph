import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# Constants and default values
SENSOR_CONFIGS = {
    "LSM6DSV": {
        "name": "LSM6DSV (IMU)",
        "typical_active": 0.695,
        "typical_sleep": 0.0015,
        "low_power_active": 0.29,
        "low_power_sleep": 0.0015
    },
    "MMC5983MA": {
        "name": "MMC5983MA (Magnetometer)",
        "typical_active": 1.0,
        "typical_sleep": 0.002,
        "low_power_active": 0.5,
        "low_power_sleep": 0.002
    },
    "TSL2591": {
        "name": "TSL2591 (Light)",
        "typical_active": 0.74,
        "typical_sleep": 0.0024,
        "low_power_active": 0.37,
        "low_power_sleep": 0.0024
    },
    "BME280": {
        "name": "BME280 (Environmental)",
        "typical_active": 0.714,
        "typical_sleep": 0.0024,
        "low_power_active": 0.35,
        "low_power_sleep": 0.0024
    }
}

COMMS_CONFIGS = {
    "GPS": {
        "name": "MAX-M10S GPS",
        "active_power": 20.0,
        "acquisition_power": 25.0,
        "default_frequency": 6.0,  # Every 10 minutes
        "default_duration": 30.0
    },
    "CELLULAR": {
        "name": "EG25-G Cellular",
        "active_power": 500.0,
        "startup_power": 600.0,
        "default_duration": 10  # minutes per day
    },
    "LORA": {
        "name": "LLCC68 LoRa",
        "active_power": 100.0,
        "listen_power": 10.0,
        "sleep_power": 0.0015
    }
}

COPROCESSOR_CONFIGS = {
    "Jetson Orin Nano": {
        "active_power": 14000.0,
        "idle_power": 400.0,
        "startup_power": 15000.0
    },
    "Raspberry Pi CM4": {
        "active_power": 7000.0,
        "idle_power": 0.0075,
        "startup_power": 8000.0
    }
}

def calculate_sensor_power(sensors_config):
    """Calculate power for sensor configuration."""
    total_active = 0
    total_sleep = 0
    
    # Process only sensor entries
    for sensor_id, sensor in sensors_config.items():
        if sensor_id not in ['base_frequency_per_hour', 'base_duration_seconds']:
            if sensor['enabled']:
                total_active += sensor['active_power']
                total_sleep += sensor['sleep_power']
            
    return total_active, total_sleep

def calculate_state_power_consumption(sensors_config, comms_config, period_hours=24):
    """Calculate power consumption for sensors and communications."""
    timeline_length = period_hours * 3600  # seconds
    power_timeline = np.zeros(timeline_length)
    
    # 1. Base sensor power
    sensor_active_power, sensor_sleep_power = calculate_sensor_power(sensors_config)
    base_frequency = sensors_config.get('base_frequency_per_hour', 60.0)
    base_duration = sensors_config.get('base_duration_seconds', 0.1)
    
    # Apply base sensor power
    base_interval = int(3600 / base_frequency)
    for t in range(0, timeline_length, base_interval):
        if t + base_duration < timeline_length:
            power_timeline[t:t + int(base_duration)] += sensor_active_power
    
    # Fill rest with sleep power
    power_timeline[power_timeline == 0] = sensor_sleep_power
    
    # 2. GPS power (if enabled)
    if comms_config['gps']['enabled']:
        gps = comms_config['gps']
        gps_interval = int(3600 / gps['frequency_per_hour'])
        
        for t in range(0, timeline_length, gps_interval):
            if t + int(gps['duration_seconds']) < timeline_length:
                # Add acquisition spike for first second
                power_timeline[t] += gps['acquisition_power']
                # Add normal GPS active power
                power_timeline[t + 1:t + int(gps['duration_seconds'])] += gps['active_power']
    
    # 3. Cellular power (if enabled)
    if comms_config['cellular']['enabled']:
        cellular = comms_config['cellular']
        start_time = cellular['start_hour'] * 3600
        end_time = start_time + (cellular['duration_minutes'] * 60)
        
        if end_time < timeline_length:
            # Add startup spike
            power_timeline[start_time] += cellular['startup_power']
            # Add active power
            power_timeline[start_time + 1:end_time] += cellular['active_power']
    
    # 4. LoRa power (if enabled)
    if comms_config['lora']['enabled']:
        lora = comms_config['lora']
        if lora['frequency_type'] == 'per_hour':
            interval = int(3600 / lora['frequency'])
        else:  # per_day
            interval = int(24 * 3600 / lora['frequency'])
        
        for t in range(0, timeline_length, interval):
            if t + int(lora['duration_seconds']) < timeline_length:
                power_timeline[t:t + int(lora['duration_seconds'])] += lora['active_power']
        
        # Add listening power if enabled
        if lora['listen_enabled']:
            power_timeline += lora['listen_power']
    
    return np.sum(power_timeline) / 3600  # Convert to mWh

def calculate_coprocessor_consumption(config, period_hours=24):
    """Calculate coprocessor power consumption."""
    if not config['enabled']:
        return 0.0
        
    total_active_minutes = (config['duration_minutes'] * 
                          (1 if config['schedule_type'] == 'daily' else config['runs_per_day']))
    active_hours = total_active_minutes / 60
    
    # Account for startup power spikes
    startup_energy = (config['startup_power'] * config['runs_per_day'] * 
                     (1/3600))  # 1-second startup spike, converted to hours
    
    # Regular operation energy
    active_energy = config['active_power'] * active_hours
    idle_energy = config['idle_power'] * (period_hours - active_hours)
    
    return startup_energy + active_energy + idle_energy

def calculate_battery_life(battery_capacity_wh, daily_consumption_mwh):
    """Calculate expected battery life in days."""
    if daily_consumption_mwh <= 0:
        return float('inf')
    battery_capacity_mwh = battery_capacity_wh * 1000
    return battery_capacity_mwh / daily_consumption_mwh

def plot_battery_life(daily_consumption_mwh):
    """Plot battery life analysis."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Calculate battery life curve
    battery_capacities = np.linspace(10, 150, 100)
    battery_lives = [calculate_battery_life(cap, daily_consumption_mwh) 
                    for cap in battery_capacities]
    
    # Plot main curve
    ax.plot(battery_capacities, battery_lives, 'b-', label='Battery Life')
    
    # Add target battery vertical lines
    target_batteries = [77.0, 100.7]
    for capacity in target_batteries:
        life = calculate_battery_life(capacity, daily_consumption_mwh)
        ax.axvline(x=capacity, color='r', linestyle='--', 
                  label=f'{capacity}Wh Battery: {life:.1f} days')
        # Add horizontal reference line
        ax.axhline(y=life, color='g', linestyle=':', alpha=0.5)
    
    # Add lifetime zones
    ax.axhspan(0, 30, alpha=0.1, color='red', label='< 30 days')
    ax.axhspan(30, 90, alpha=0.1, color='yellow', label='30-90 days')
    ax.axhspan(90, max(battery_lives), alpha=0.1, color='green', label='> 90 days')
    
    ax.set_xlabel('Battery Capacity (Wh)')
    ax.set_ylabel('Battery Life (Days)')
    ax.set_title('Battery Life vs. Capacity')
    ax.grid(True)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    return fig

# Streamlit UI
st.title("Battery Life Calculator")

# 1. SENSOR CONFIGURATION
st.header("Sensor Configuration")
st.markdown("---")

# Power mode selection
sensor_power_mode = st.radio("Select Sensor Power Mode:", ["Typical", "Low Power"])

# Initialize sensor configuration
sensors_config = {}

with st.expander("Individual Sensor Configuration"):
    # Add each sensor
    for sensor_id, sensor in SENSOR_CONFIGS.items():
        enabled = st.checkbox(f"Enable {sensor['name']}", value=True)
        
        if sensor_power_mode == "Typical":
            active_power = sensor["typical_active"]
            sleep_power = sensor["typical_sleep"]
        else:
            active_power = sensor["low_power_active"]
            sleep_power = sensor["low_power_sleep"]
        
        sensors_config[sensor_id] = {
            "enabled": enabled,
            "active_power": active_power,
            "sleep_power": sleep_power
        }
    
    # Sensor timing
    st.write("\nBase Sensor Timing:")
    sensors_config['base_frequency_per_hour'] = st.number_input(
        "Base Sampling Frequency (times per hour):",
        value=60.0,
        min_value=1.0
    )
    sensors_config['base_duration_seconds'] = st.number_input(
        "Sampling Duration (seconds):",
        value=0.1,
        min_value=0.1
    )

# 2. COMMUNICATIONS CONFIGURATION
st.header("Communications Configuration")
st.markdown("---")

# GPS Configuration
with st.expander("GPS Configuration (MAX-M10S)"):
    gps_enabled = st.checkbox("Enable GPS", value=True)
    gps_config = {
        "enabled": gps_enabled,
        "active_power": COMMS_CONFIGS["GPS"]["active_power"],
        "acquisition_power": COMMS_CONFIGS["GPS"]["acquisition_power"],
        "frequency_per_hour": st.number_input(
            "GPS Update Frequency (times per hour):",
            value=COMMS_CONFIGS["GPS"]["default_frequency"],
            min_value=0.1
        ),
        "duration_seconds": st.number_input(
            "GPS Fix Duration (seconds):",
            value=COMMS_CONFIGS["GPS"]["default_duration"],
            min_value=0.1
        )
    }

# Cellular Configuration
with st.expander("Cellular Configuration (EG25-G)"):
    cellular_enabled = st.checkbox("Enable Cellular", value=True)
    cellular_config = {
        "enabled": cellular_enabled,
        "active_power": COMMS_CONFIGS["CELLULAR"]["active_power"],
        "startup_power": COMMS_CONFIGS["CELLULAR"]["startup_power"],
        "start_hour": st.number_input(
            "Start Hour (0-23):",
            value=1,
            min_value=0,
            max_value=23
        ),
        "duration_minutes": st.number_input(
            "Daily Communication Window (minutes):",
            value=COMMS_CONFIGS["CELLULAR"]["default_duration"],
            min_value=1
        )
    }

# LoRa Configuration
with st.expander("LoRa Configuration (LLCC68)"):
    lora_enabled = st.checkbox("Enable LoRa", value=True)
    lora_config = {
        "enabled": lora_enabled,
        "active_power": COMMS_CONFIGS["LORA"]["active_power"],
        "listen_power": COMMS_CONFIGS["LORA"]["listen_power"],
        "sleep_power": COMMS_CONFIGS["LORA"]["sleep_power"],
        "listen_enabled": st.checkbox("Enable Listen Mode", value=False),
        "frequency_type": st.selectbox(
            "Message Frequency Type:",
            ["per_hour", "per_day"]
        ),
        "frequency": st.number_input(
            "Message Frequency (per selected period):",
            value=1.0,
            min_value=0.1
        ),
        "duration_seconds": st.number_input(
            "Message Duration (seconds):",
            value=5.0,
            min_value=0.1
        )
    }

# Combine all comms config
comms_config = {
    "gps": gps_config,
    "cellular": cellular_config,
    "lora": lora_config
}

# 3. COPROCESSOR CONFIGURATION
st.header("Co-Processor Configuration")
st.markdown("---")

coprocessor_enabled = st.checkbox("Enable Co-Processor", value=False)

if coprocessor_enabled:
    with st.expander("Co-Processor Configuration"):
        coprocessor_type = st.selectbox(
            "Select Co-Processor Type:",
            list(COPROCESSOR_CONFIGS.keys())
        )
        
        coprocessor_config = {
            "enabled": True,
            "active_power": COPROCESSOR_CONFIGS[coprocessor_type]["active_power"],
            "idle_power": COPROCESSOR_CONFIGS[coprocessor_type]["idle_power"],
            "startup_power": COPROCESSOR_CONFIGS[coprocessor_type]["startup_power"],
            "schedule_type": st.radio(
                "Schedule Type:",
                ["daily", "interval"]
            )
        }
        
        if coprocessor_config["schedule_type"] == "daily":
            coprocessor_config.update({
                "duration_minutes": st.number_input(
                    "Processing Window Duration (minutes):",
                    value=30,
                    min_value=1
                ),
                "runs_per_day": 1
            })
        else:
            coprocessor_config.update({
                "runs_per_day": st.number_input(
                    "Number of Processing Windows per Day:",
                    value=4,
                    min_value=1
                ),
                "duration_minutes": st.number_input(
                    "Duration per Window (minutes):",
                    value=5,
                    min_value=1
                )
            })
else:
    coprocessor_config = {"enabled": False}

# 4. BATTERY CONFIGURATION
st.header("Battery Configuration")
st.markdown("---")

with st.expander("Battery Details"):
    temperature_derating = st.slider(
        "Temperature Derating Factor (%)", 
        min_value=70, 
        max_value=100, 
        value=85,
        help="Adjust for temperature effects on battery capacity"
    )
    
    aging_factor = st.slider(
        "Battery Aging Factor (%)", 
        min_value=70, 
        max_value=100, 
        value=90,
        help="Account for battery capacity degradation over time"
    )
    
    voltage_efficiency = st.slider(
        "Voltage Converter Efficiency (%)",
        min_value=80,
        max_value=95,
        value=85,
        help="Account for voltage converter losses"
    )

# 5. CALCULATION AND RESULTS
if st.button("Calculate Battery Life"):
    # Calculate base consumption
    base_consumption = calculate_state_power_consumption(sensors_config, comms_config)
    
    # Add coprocessor consumption if enabled
    coprocessor_consumption = calculate_coprocessor_consumption(coprocessor_config) if coprocessor_config['enabled'] else 0
    
    # Total consumption before derating
    total_daily_consumption = base_consumption + coprocessor_consumption
    
    # Apply efficiency factors
    efficiency_factor = (temperature_derating / 100) * (aging_factor / 100) * (voltage_efficiency / 100)
    derated_daily_consumption = total_daily_consumption / efficiency_factor
    
    # Display Results
    st.subheader("Power Consumption Breakdown")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("Daily Energy Consumption:")
        st.write(f"Base System: {base_consumption:.2f} mWh")
        if coprocessor_config['enabled']:
            st.write(f"Co-Processor: {coprocessor_consumption:.2f} mWh")
        st.write(f"Total (before derating): {total_daily_consumption:.2f} mWh")
        st.write(f"Total (after derating): {derated_daily_consumption:.2f} mWh")
    
    with col2:
        st.write("Average Power Draw:")
        st.write(f"Base System: {(base_consumption/24):.2f} mW")
        if coprocessor_config['enabled']:
            st.write(f"Co-Processor: {(coprocessor_consumption/24):.2f} mW")
        st.write(f"Total (before derating): {(total_daily_consumption/24):.2f} mW")
        st.write(f"Total (after derating): {(derated_daily_consumption/24):.2f} mW")
    
    st.subheader("Efficiency Factors")
    st.write(f"Temperature Derating: {temperature_derating}%")
    st.write(f"Aging Factor: {aging_factor}%")
    st.write(f"Voltage Efficiency: {voltage_efficiency}%")
    st.write(f"Combined Efficiency: {efficiency_factor*100:.1f}%")
    
    st.subheader("Battery Life Estimates")
    # Calculate for both battery capacities
    for capacity in [77.0, 100.7]:
        battery_life = calculate_battery_life(capacity, derated_daily_consumption)
        months = battery_life / 30.44  # Average days per month
        st.write(f"\n{capacity}Wh Battery:")
        st.write(f"- {battery_life:.1f} days")
        st.write(f"- {months:.1f} months")
        
        # Add warnings based on target lifetime
        if battery_life < 30:
            st.warning(f"⚠️ Battery life is less than 30 days!")
        elif battery_life < 90:
            st.info("ℹ️ Battery life is between 30-90 days")
        else:
            st.success("✅ Battery life exceeds 90 days")
    
    # Create and display the battery life plot
    fig = plot_battery_life(derated_daily_consumption)
    st.pyplot(fig)
    
    # Additional Analysis
    st.subheader("Detailed Analysis")
    
    # Power Budget Analysis
    st.write("Power Budget Breakdown:")
    total_power = derated_daily_consumption/24
    if total_power > 0:
        components = [
            ("Sensors (Base)", (base_consumption/24) * (sensor_active_power/(sensor_active_power + sensor_sleep_power))),
            ("Sensors (Sleep)", (base_consumption/24) * (sensor_sleep_power/(sensor_active_power + sensor_sleep_power))),
        ]
        if gps_config['enabled']:
            components.append(("GPS", (gps_config['active_power'] * gps_config['duration_seconds'] * 
                                    gps_config['frequency_per_hour']) / 3600))
        if cellular_config['enabled']:
            components.append(("Cellular", (cellular_config['active_power'] * cellular_config['duration_minutes']) / 
                                        (24 * 60)))
        if lora_config['enabled']:
            components.append(("LoRa", (lora_config['active_power'] * lora_config['duration_seconds'] * 
                                     (24 if lora_config['frequency_type'] == 'per_hour' else 1) * 
                                     lora_config['frequency']) / (24 * 3600)))
        if coprocessor_config['enabled']:
            components.append(("Co-Processor", coprocessor_consumption/24))
        
        # Create pie chart
        fig_pie, ax_pie = plt.subplots()
        labels = [c[0] for c in components]
        sizes = [c[1] for c in components]
        ax_pie.pie(sizes, labels=labels, autopct='%1.1f%%')
        ax_pie.axis('equal')
        st.pyplot(fig_pie)
        
        # Show recommendations based on analysis
        st.subheader("Recommendations")
        max_component = max(components, key=lambda x: x[1])
        if max_component[1]/total_power > 0.5:  # If any component uses more than 50% of power
            st.warning(f"{max_component[0]} is consuming {max_component[1]/total_power*100:.1f}% of total power. "
                      f"Consider optimizing this component first.")
        
        # Suggest optimizations based on configuration
        if coprocessor_config['enabled'] and coprocessor_consumption/24 > 1000:  # If coprocessor using more than 1W
            st.info("Consider reducing co-processor active time or using low-power mode when possible.")
        
        if lora_config['listen_enabled']:
            st.info("LoRa listen mode significantly impacts battery life. Consider using a more aggressive sleep cycle.")
            
        if cellular_config['enabled'] and cellular_config['duration_minutes'] > 15:
            st.info("Consider reducing cellular communication window to improve battery life.")

    # Save results to session state for potential export
    st.session_state.last_calculation = {
        "daily_consumption": derated_daily_consumption,
        "battery_life_77Wh": calculate_battery_life(77.0, derated_daily_consumption),
        "battery_life_100Wh": calculate_battery_life(100.7, derated_daily_consumption),
        "average_power": derated_daily_consumption/24
    }

# Add export functionality
if 'last_calculation' in st.session_state:
    if st.button("Export Results"):
        results_str = f"""Battery Life Calculation Results
        
Daily Consumption: {st.session_state.last_calculation['daily_consumption']:.2f} mWh
Average Power Draw: {st.session_state.last_calculation['average_power']:.2f} mW
Battery Life (77Wh): {st.session_state.last_calculation['battery_life_77Wh']:.1f} days
Battery Life (100.7Wh): {st.session_state.last_calculation['battery_life_100Wh']:.1f} days
"""
        st.download_button(
            label="Download Results",
            data=results_str,
            file_name="battery_calculation_results.txt",
            mime="text/plain"
        )