import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

def calculate_power_consumption(active_power, sleep_power, process_time_sec, duty_cycle_per_day):
    """
    Calculate daily power consumption based on process time and duty cycle.
    Returns consumption in mWh.
    """
    # Convert process time to hours
    process_time_hr = process_time_sec / 3600
    
    # Calculate total active time per day
    total_active_time_hr = process_time_hr * duty_cycle_per_day
    
    # Calculate power consumption
    active_consumption = active_power * total_active_time_hr
    sleep_consumption = sleep_power * (24 - total_active_time_hr)
    
    return active_consumption + sleep_consumption

def calculate_battery_life(battery_capacity_wh, daily_consumption_mwh):
    """Calculate expected battery life in days."""
    battery_capacity_mwh = battery_capacity_wh * 1000
    return battery_capacity_mwh / daily_consumption_mwh if daily_consumption_mwh > 0 else 0

def plot_battery_life(daily_consumption_mwh):
    """Plot the projected battery life against different battery capacities."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Define battery capacities
    battery_capacities = np.linspace(10, 150, 100)  # Range of capacities to plot
    battery_lives = [calculate_battery_life(cap, daily_consumption_mwh) for cap in battery_capacities]
    
    # Plot battery life curve
    ax.plot(battery_capacities, battery_lives, 'b-', label='Battery Life')
    
    # Add vertical lines for target batteries
    target_batteries = [77.0, 100.7]
    for capacity in target_batteries:
        life = calculate_battery_life(capacity, daily_consumption_mwh)
        ax.axvline(x=capacity, color='r', linestyle='--', 
                  label=f'{capacity}Wh Battery: {life:.1f} days')
        
        # Add horizontal line at target life for reference
        ax.axhline(y=life, color='g', linestyle=':', alpha=0.5)
    
    ax.set_xlabel('Battery Capacity (Wh)')
    ax.set_ylabel('Battery Life (Days)')
    ax.set_title('Battery Life vs. Capacity')
    ax.grid(True)
    ax.legend()
    
    # Add efficiency warning zone
    ax.axhspan(0, 30, alpha=0.1, color='red', label='< 30 days')
    ax.axhspan(30, 90, alpha=0.1, color='yellow', label='30-90 days')
    ax.axhspan(90, max(battery_lives), alpha=0.1, color='green', label='> 90 days')
    
    return fig

# Streamlit GUI
st.title("Battery Life Calculator")

# MCU SECTION
st.header("MCU Configuration")
st.markdown("---")

# Basic MCU selection
mcu_type = st.selectbox("Select MCU Configuration:", [
    "Sensors + STM32 (Max)",
    "Sensors + STM32 (Typical)",
    "Sensors + STM32 (Low Powered)",
    "Without Comms (Max)",
    "Without Comms (Typical)",
    "Without Comms (Low Powered)",
    "Without Comms/GPS (Max)",
    "Without Comms/GPS (Typical)",
    "Without Comms/GPS (Low Powered)"
])

# Expander for detailed MCU power configuration
with st.expander("MCU Power Configuration Details"):
    mcu_power_mapping = {
        "Sensors + STM32 (Max)": {"active": 556.244, "sleep": 6.216903},
        "Sensors + STM32 (Typical)": {"active": 161.74, "sleep": 6.216903},
        "Sensors + STM32 (Low Powered)": {"active": 88.15, "sleep": 6.216903},
        "Without Comms (Max)": {"active": 60.2547, "sleep": 0.276903},
        "Without Comms (Typical)": {"active": 41.95488, "sleep": 0.276903},
        "Without Comms (Low Powered)": {"active": 36.46516, "sleep": 0.276903},
        "Without Comms/GPS (Max)": {"active": 17.3547, "sleep": 0.125103},
        "Without Comms/GPS (Typical)": {"active": 8.95488, "sleep": 0.125103},
        "Without Comms/GPS (Low Powered)": {"active": 3.46516, "sleep": 0.125103}
    }
    
    # Allow modification of default values
    mcu_active_power = st.number_input(
        "MCU Active Power (mW):", 
        value=float(mcu_power_mapping[mcu_type]["active"]),
        step=0.000001,
        format="%.6f"
    )
    mcu_sleep_power = st.number_input(
        "MCU Sleep Power (mW):", 
        value=float(mcu_power_mapping[mcu_type]["sleep"]),
        step=0.000001,
        format="%.6f"
    )

# MCU timing configuration
with st.expander("MCU Timing Configuration"):
    mcu_process_time = st.number_input(
        "Process Time per Run (seconds):",
        min_value=0.1,
        value=10.0,
        help="How long does each process run take?"
    )
    mcu_duty_cycle = st.number_input(
        "Number of Runs per Day:",
        min_value=1,
        value=24,
        help="How many times per day does the process run?"
    )
    
    # Display effective duty cycle
    effective_duty_cycle = (mcu_process_time * mcu_duty_cycle) / (24 * 3600) * 100
    st.write(f"Effective Duty Cycle: {effective_duty_cycle:.4f}%")

# COPROCESSOR SECTION
st.header("Co-Processor Configuration")
st.markdown("---")

coprocessor_type = st.selectbox("Select Co-Processor:", [
    "None",
    "Jetson Orin Nano (High Power)",
    "Jetson Orin Nano (Low Power)",
    "CM4 (High Power)",
    "CM4 (Low Power)"
])

# Expander for detailed coprocessor configuration
with st.expander("Co-Processor Power Configuration Details"):
    coprocessor_power_mapping = {
        "None": {"active": 0.0, "idle": 0.0},
        "Jetson Orin Nano (High Power)": {"active": 14000.0, "idle": 7000.0},
        "Jetson Orin Nano (Low Power)": {"active": 7000.0, "idle": 400.0},
        "CM4 (High Power)": {"active": 7000.0, "idle": 2000.0},
        "CM4 (Low Power)": {"active": 2000.0, "idle": 0.0075}
    }
    
    if coprocessor_type != "None":
        coprocessor_active_power = st.number_input(
            "Co-Processor Active Power (mW):",
            value=float(coprocessor_power_mapping[coprocessor_type]["active"]),
            step=0.000001,
            format="%.6f"
        )
        coprocessor_idle_power = st.number_input(
            "Co-Processor Idle Power (mW):",
            value=float(coprocessor_power_mapping[coprocessor_type]["idle"]),
            step=0.000001,
            format="%.6f"
        )
    else:
        coprocessor_active_power = 0.0
        coprocessor_idle_power = 0.0

# Coprocessor timing configuration
with st.expander("Co-Processor Timing Configuration"):
    if coprocessor_type != "None":
        coprocessor_process_time = st.number_input(
            "Co-Processor Process Time per Run (seconds):",
            min_value=0.1,
            value=10.0,
            help="How long does each co-processor task take?"
        )
        coprocessor_duty_cycle = st.number_input(
            "Co-Processor Runs per Day:",
            min_value=1,
            value=24,
            help="How many times per day does the co-processor run?"
        )
        
        # Display effective duty cycle
        coprocessor_effective_duty_cycle = (coprocessor_process_time * coprocessor_duty_cycle) / (24 * 3600) * 100
        st.write(f"Effective Duty Cycle: {coprocessor_effective_duty_cycle:.4f}%")
    else:
        coprocessor_process_time = 0
        coprocessor_duty_cycle = 0

# BATTERY CONFIGURATION
st.header("Battery Configuration")
st.markdown("---")

with st.expander("Battery Details"):
    # Add temperature derating factor
    temperature_derating = st.slider(
        "Temperature Derating Factor (%)", 
        min_value=70, 
        max_value=100, 
        value=85,
        help="Adjust for temperature effects on battery capacity"
    )
    
    # Add aging factor
    aging_factor = st.slider(
        "Battery Aging Factor (%)", 
        min_value=70, 
        max_value=100, 
        value=90,
        help="Account for battery capacity degradation over time"
    )

# CALCULATION SECTION
st.header("Battery Analysis")
st.markdown("---")

# Calculate Button
if st.button("Calculate Battery Life"):
    # Calculate MCU consumption
    mcu_daily_consumption = calculate_power_consumption(
        mcu_active_power,
        mcu_sleep_power,
        mcu_process_time,
        mcu_duty_cycle
    )
    
    # Calculate Coprocessor consumption
    coprocessor_daily_consumption = calculate_power_consumption(
        coprocessor_active_power,
        coprocessor_idle_power,
        coprocessor_process_time,
        coprocessor_duty_cycle
    )
    
    total_daily_consumption = mcu_daily_consumption + coprocessor_daily_consumption
    
    # Apply derating factors
    effective_capacity_factor = (temperature_derating / 100) * (aging_factor / 100)
    derated_daily_consumption = total_daily_consumption / effective_capacity_factor
    
    st.session_state.daily_consumption_mwh = derated_daily_consumption
    
    # Display results
    col1, col2 = st.columns(2)
    with col1:
        st.write("Daily Consumption Breakdown:")
        st.write(f"MCU: {mcu_daily_consumption:.2f} mWh")
        st.write(f"Co-Processor: {coprocessor_daily_consumption:.2f} mWh")
        st.write(f"Total (before derating): {total_daily_consumption:.2f} mWh")
        st.write(f"Total (after derating): {derated_daily_consumption:.2f} mWh")
    
    with col2:
        st.write("Average Power Draw:")
        st.write(f"MCU: {(mcu_daily_consumption/24):.2f} mW")
        st.write(f"Co-Processor: {(coprocessor_daily_consumption/24):.2f} mW")
        st.write(f"Total: {(derated_daily_consumption/24):.2f} mW")
    
    # Calculate and display battery life for both capacities
    st.write("\nBattery Life Estimates:")
    for capacity in [77.0, 100.7]:
        battery_life = calculate_battery_life(capacity, derated_daily_consumption)
        st.write(f"{capacity}Wh Battery: {battery_life:.2f} days")
    
    # Plot the results
    fig = plot_battery_life(derated_daily_consumption)
    st.pyplot(fig)