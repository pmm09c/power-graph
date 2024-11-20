import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# =============================================================================
# CONSTANTS AND DEFAULT VALUES
# =============================================================================

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
        "default_frequency": 6.0,  # Every 10 minutes
        "default_duration": 30.0   # seconds
    },
    "CELLULAR": {
        "name": "EG25-G Cellular",
        "active_power": 500.0,
        "default_duration": 10,    # minutes per day
        "frequency": 1            # once per day
    },
    "LORA": {
        "name": "LLCC68 LoRa",
        "active_power": 100.0,
        "default_duration": 5.0   # seconds
    }
}

COPROCESSOR_CONFIGS = {
    "Jetson Orin Nano": {
        "active_power": 14000.0,
        "idle_power": 400.0
    },
    "Raspberry Pi CM4": {
        "active_power": 7000.0,
        "idle_power": 0.0075
    }
}

# =============================================================================
# CALCULATION FUNCTIONS
# =============================================================================

def calculate_sensor_power(sensors_config):
    """Calculate power for sensor configuration."""
    total_active = 0
    total_sleep = 0
    
    for sensor_id, sensor in sensors_config.items():
        if sensor_id not in ['base_frequency_per_hour', 'base_duration_seconds']:
            if sensor['enabled']:
                total_active += sensor['active_power']
                total_sleep += sensor['sleep_power']
            
    return total_active, total_sleep

def calculate_state_power_consumption(sensors_config, comms_config):
    """Calculate power consumption for sensors and communications."""
    # Initialize total daily consumption
    total_daily_consumption = 0
    
    # 1. Calculate Sensor Power
    sensor_active_power, sensor_sleep_power = calculate_sensor_power(sensors_config)
    base_frequency = sensors_config.get('base_frequency_per_hour', 60.0)
    base_duration = sensors_config.get('base_duration_seconds', 0.1)
    
    # Calculate daily sensor energy
    active_hours = (base_frequency * base_duration * 24) / 3600  # Convert to hours
    sleep_hours = 24 - active_hours
    
    sensor_daily_consumption = (sensor_active_power * active_hours + 
                              sensor_sleep_power * sleep_hours)
    total_daily_consumption += sensor_daily_consumption
    
    # 2. Add GPS Power
    if comms_config['gps']['enabled']:
        gps = comms_config['gps']
        gps_active_hours = (gps['duration_seconds'] * gps['frequency_per_hour'] * 24) / 3600
        gps_power = gps['active_power'] * gps_active_hours
        total_daily_consumption += gps_power
    
    # 3. Add Cellular Power
    if comms_config['cellular']['enabled']:
        cellular = comms_config['cellular']
        cellular_active_hours = (cellular['duration_minutes'] * 
                               cellular['frequency_per_day']) / 60
        cellular_power = cellular['active_power'] * cellular_active_hours
        total_daily_consumption += cellular_power
    
    # 4. Add LoRa Power
    if comms_config['lora']['enabled']:
        lora = comms_config['lora']
        messages_per_day = (24 if lora['frequency_type'] == 'per_hour' 
                          else 1) * lora['frequency']
        lora_active_hours = (lora['duration_seconds'] * messages_per_day) / 3600
        lora_power = lora['active_power'] * lora_active_hours
        total_daily_consumption += lora_power
    
    return total_daily_consumption

def calculate_coprocessor_consumption(config):
    """Calculate coprocessor power consumption."""
    if not config['enabled']:
        return 0.0
    
    active_hours = (config['duration_minutes'] * config['frequency_per_day']) / 60
    idle_hours = 24 - active_hours
    
    active_power = config['active_power'] * active_hours
    idle_power = config['idle_power'] * idle_hours
    
    return active_power + idle_power

def calculate_battery_life(battery_capacity_wh, daily_consumption_mwh):
    """Calculate expected battery life in days."""
    if daily_consumption_mwh <= 0:
        return float('inf')
    battery_capacity_mwh = battery_capacity_wh * 1000
    return battery_capacity_mwh / daily_consumption_mwh

def plot_battery_life(daily_consumption_mwh):
    """Plot battery life vs capacity."""
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
    ax.legend()
    
    return fig

# =============================================================================
# STREAMLIT UI
# =============================================================================

st.title("Battery Life Calculator")

# SENSOR CONFIGURATION
st.header("Sensor Configuration")
st.markdown("---")

sensor_power_mode = st.radio("Select Sensor Power Mode:", ["Typical", "Low Power"])

sensors_config = {}
with st.expander("Individual Sensor Configuration"):
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

# COMMUNICATIONS CONFIGURATION
st.header("Communications Configuration")
st.markdown("---")

# GPS Configuration
with st.expander("GPS Configuration (MAX-M10S)"):
    gps_enabled = st.checkbox("Enable GPS", value=True)
    if gps_enabled:
        gps_config = {
            "enabled": True,
            "active_power": st.number_input(
                "GPS Active Power (mW):",
                value=COMMS_CONFIGS["GPS"]["active_power"],
                format="%.2f"
            ),
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
    else:
        gps_config = {"enabled": False}

# Cellular Configuration
with st.expander("Cellular Configuration (EG25-G)"):
    cellular_enabled = st.checkbox("Enable Cellular (EG25-G)", value=True)
    if cellular_enabled:
        cellular_config = {
            "enabled": True,
            "active_power": st.number_input(
                "Cellular Active Power (mW):",
                value=COMMS_CONFIGS["CELLULAR"]["active_power"],
                format="%.2f"
            ),
            "frequency_per_day": st.number_input(
                "Updates per Day:",
                value=1,
                min_value=1,
                help="How many times per day the cellular modem activates"
            ),
            "duration_minutes": st.number_input(
                "Active Duration (minutes):",
                value=COMMS_CONFIGS["CELLULAR"]["default_duration"],
                min_value=1,
                help="How long the cellular modem is active each time"
            )
        }
    else:
        cellular_config = {"enabled": False}

# LoRa Configuration
with st.expander("LoRa Configuration (LLCC68)"):
    lora_enabled = st.checkbox("Enable LoRa", value=True)
    if lora_enabled:
        lora_config = {
            "enabled": True,
            "active_power": st.number_input(
                "LoRa Active Power (mW):",
                value=COMMS_CONFIGS["LORA"]["active_power"],
                format="%.2f"
            ),
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
                value=COMMS_CONFIGS["LORA"]["default_duration"],
                min_value=0.1
            )
        }
    else:
        lora_config = {"enabled": False}

# Combine all comms config
comms_config = {
    "gps": gps_config,
    "cellular": cellular_config,
    "lora": lora_config
}

# COPROCESSOR CONFIGURATION
st.header("Co-Processor Configuration")
st.markdown("---")

coprocessor_enabled = st.checkbox("Enable Co-Processor", value=False)
if coprocessor_enabled:
    coprocessor_type = st.selectbox(
        "Select Co-Processor Type:",
        list(COPROCESSOR_CONFIGS.keys())
    )
    
    coprocessor_config = {
        "enabled": True,
        "active_power": COPROCESSOR_CONFIGS[coprocessor_type]["active_power"],
        "idle_power": COPROCESSOR_CONFIGS[coprocessor_type]["idle_power"],
        "frequency_per_day": st.number_input(
            "Processing Windows per Day:",
            value=1,
            min_value=1
        ),
        "duration_minutes": st.number_input(
            "Duration per Window (minutes):",
            value=5,
            min_value=1
        )
    }
else:
    coprocessor_config = {"enabled": False}

# BATTERY CONFIGURATION
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

# CALCULATION
if st.button("Calculate Battery Life"):
    # Calculate base consumption
    base_consumption = calculate_state_power_consumption(sensors_config, comms_config)
    
    # Add coprocessor consumption
    coprocessor_consumption = calculate_coprocessor_consumption(coprocessor_config)
    
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
    
    # Display efficiency factors
    st.subheader("Efficiency Factors")
    st.write(f"Temperature Derating: {temperature_derating}%")
    st.write(f"Aging Factor: {aging_factor}%")
    st.write(f"Voltage Efficiency: {voltage_efficiency}%")
    st.write(f"Combined Efficiency: {efficiency_factor*100:.1f}%")
    
    # Battery life calculations
    st.subheader("Battery Life Estimates")
    for capacity in [77.0, 100.7]:
        battery_life = calculate_battery_life(capacity, derated_daily_consumption)
        months = battery_life / 30.44  # Average days per month
        st.write(f"\n{capacity}Wh Battery:")
        st.write(f"- {battery_life:.1f} days")
        st.write(f"- {months:.1f} months")
        
        # Add lifetime indicators
        if battery_life < 30:
            st.warning(f"⚠️ Battery life is less than 30 days!")
        elif battery_life < 90:
            st.info("ℹ️ Battery life is between 30-90 days")
        else:
            st.success("✅ Battery life exceeds 90 days")
    
    # Plot battery life
    fig = plot_battery_life(derated_daily_consumption)
    st.pyplot(fig)
    
    # Component Analysis
    st.subheader("Component Power Analysis")
    
    # Calculate component percentages
    components = []
    
    # Add sensor components
    sensor_active_power, sensor_sleep_power = calculate_sensor_power(sensors_config)
    sensor_total = (base_consumption/24)
    if sensor_total > 0:
        components.append(("Sensors", sensor_total))
    
    # Add GPS if enabled
    if comms_config['gps']['enabled']:
        gps_power = (gps_config['active_power'] * gps_config['duration_seconds'] * 
                    gps_config['frequency_per_hour']) / 3600
        if gps_power > 0:
            components.append(("GPS", gps_power))
    
    # Add Cellular if enabled
    if comms_config['cellular']['enabled']:
        cellular_power = (cellular_config['active_power'] * cellular_config['duration_minutes'] * 
                        cellular_config['frequency_per_day']) / (24 * 60)
        if cellular_power > 0:
            components.append(("Cellular", cellular_power))
    
    # Add LoRa if enabled
    if comms_config['lora']['enabled']:
        lora_power = (lora_config['active_power'] * lora_config['duration_seconds'] * 
                     (24 if lora_config['frequency_type'] == 'per_hour' else 1) * 
                     lora_config['frequency']) / (24 * 3600)
        if lora_power > 0:
            components.append(("LoRa", lora_power))
    
    # Add Coprocessor if enabled
    if coprocessor_config['enabled']:
        coprocessor_power = coprocessor_consumption/24
        if coprocessor_power > 0:
            components.append(("Co-Processor", coprocessor_power))
    
    # Create pie chart of power distribution
    if components:
        fig_pie, ax_pie = plt.subplots()
        labels = [c[0] for c in components]
        sizes = [c[1] for c in components]
        
        # Filter out very small values for better visualization
        min_percentage = 1.0  # minimum percentage to show in pie chart
        total = sum(sizes)
        filtered_components = [(label, size) for label, size in zip(labels, sizes) 
                             if (size/total)*100 >= min_percentage]
        
        if filtered_components:
            pie_labels = [c[0] for c in filtered_components]
            pie_sizes = [c[1] for c in filtered_components]
            
            ax_pie.pie(pie_sizes, labels=pie_labels, autopct='%1.1f%%')
            ax_pie.axis('equal')
            st.pyplot(fig_pie)
    
    # Power Budget Recommendations
    st.subheader("Power Optimization Recommendations")
    
    total_power = derated_daily_consumption/24
    if total_power > 0:
        # Find highest power consumer
        max_component = max(components, key=lambda x: x[1])
        if max_component[1]/total_power > 0.5:
            st.warning(f"{max_component[0]} consumes {max_component[1]/total_power*100:.1f}% of total power. "
                      f"Consider optimizing this component first.")
        
        # Specific recommendations
        if coprocessor_config['enabled'] and coprocessor_consumption/24 > 1000:
            st.info("Consider reducing co-processor active time or using lower power states.")
            
        if comms_config['cellular']['enabled'] and cellular_config['duration_minutes'] > 15:
            st.info("Consider reducing cellular communication window duration.")
            
        if comms_config['gps']['enabled'] and gps_config['frequency_per_hour'] > 6:
            st.info("Consider reducing GPS update frequency to save power.")

    # Save calculation results to session state
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