import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# =============================================================================
# CONSTANTS AND DEFAULT VALUES
# =============================================================================

SENSOR_CONFIGS = {
    "LSM6DSV": {
        "name": "LSM6DSV (IMU)",
        "description": "Always-on motion sensing",
        "operation_mode": "continuous",  # New: indicates always-on operation
        "typical": {
            "active": 0.695,  # mW
            "sleep": 0.0015   # mW
        },
        "low_power": {
            "active": 0.29,   # mW
            "sleep": 0.0015   # mW
        },
        "sampling_rates": {   # New: available sampling rates
            "typical": 104,   # Hz
            "low_power": 52   # Hz
        }
    },
    "MMC5983MA": {
        "name": "MMC5983MA (Magnetometer)",
        "description": "Magnetic field sensing",
        "operation_mode": "polled",  # New: indicates periodic sampling
        "typical": {
            "active": 1.0,
            "sleep": 0.002
        },
        "low_power": {
            "active": 0.5,
            "sleep": 0.002
        },
        "min_sampling_period": 0.1  # seconds
    },
    "TSL2591": {
        "name": "TSL2591 (Light)",
        "description": "Ambient light sensing",
        "operation_mode": "polled",
        "typical": {
            "active": 0.74,
            "sleep": 0.0024
        },
        "low_power": {
            "active": 0.37,
            "sleep": 0.0024
        },
        "min_sampling_period": 0.1  # seconds
    },
    "BME280": {
        "name": "BME280 (Environmental)",
        "description": "Temperature, humidity, pressure",
        "operation_mode": "polled",
        "typical": {
            "active": 0.714,
            "sleep": 0.0024
        },
        "low_power": {
            "active": 0.35,
            "sleep": 0.0024
        },
        "min_sampling_period": 0.1  # seconds
    }
}

COMMS_CONFIGS = {
    "GPS": {
        "name": "MAX-M10S GPS",
        "description": "Position tracking",
        "power_modes": {
            "acquisition": {
                "power": 25.0,    # mW during initial acquisition
                "typical_duration": 30.0  # seconds for cold start
            },
            "tracking": {
                "power": 20.0,    # mW during normal tracking
                "typical_duration": 5.0   # seconds for position update
            }
        },
        "default_schedule": {
            "frequency": 6.0,     # times per hour
            "duration": 30.0      # seconds
        }
    },
    "CELLULAR": {
        "name": "EG25-G Cellular",
        "description": "Cellular communication",
        "power_modes": {
            "startup": {
                "power": 600.0,   # mW during startup
                "duration": 5.0   # seconds
            },
            "active": {
                "power": 500.0,   # mW during transmission
                "min_duration": 60.0  # minimum session in seconds
            },
            "idle": {
                "power": 100.0    # mW when registered but not transmitting
            }
        },
        "default_schedule": {
            "frequency": 1,       # times per day
            "duration": 10        # minutes
        }
    },
    "LORA": {
        "name": "LLCC68 LoRa",
        "description": "Long-range mesh communication",
        "power_modes": {
            "tx": {
                "power": 100.0,   # mW during transmission
                "typical_duration": 5.0  # seconds
            },
            "rx": {
                "power": 10.0,    # mW during receive
                "typical_duration": 2.0  # seconds
            },
            "sleep": {
                "power": 0.0015   # mW in sleep
            }
        },
        "default_schedule": {
            "tx_frequency": 1,    # times per hour
            "rx_duty_cycle": 0.1  # 10% listening time
        }
    }
}

COPROCESSOR_CONFIGS = {
    "Jetson Orin Nano": {
        "name": "NVIDIA Jetson Orin Nano",
        "description": "AI/ML processing",
        "power_modes": {
            "max": {
                "active": 14000.0,
                "idle": 400.0
            },
            "typical": {
                "active": 7000.0,
                "idle": 400.0
            }
        },
        "startup_time": 30,  # seconds
        "min_processing_window": 60  # seconds
    },
    "Raspberry Pi CM4": {
        "name": "Raspberry Pi Compute Module 4",
        "description": "General processing",
        "power_modes": {
            "max": {
                "active": 7000.0,
                "idle": 0.0075
            },
            "typical": {
                "active": 4000.0,
                "idle": 0.0075
            }
        },
        "startup_time": 20,  # seconds
        "min_processing_window": 30  # seconds
    }
}

# Battery configuration defaults
BATTERY_CONFIGS = {
    "standard": {
        "capacity": 77.0,    # Wh
        "chemistry": "Li-ion",
        "typical_derating": {
            "temperature": 85,  # %
            "aging": 90,       # %
            "voltage": 85      # %
        }
    },
    "extended": {
        "capacity": 100.7,   # Wh
        "chemistry": "Li-ion",
        "typical_derating": {
            "temperature": 85,  # %
            "aging": 90,       # %
            "voltage": 85      # %
        }
    }
}

# System operational modes
SYSTEM_MODES = {
    "CONTINUOUS": {
        "name": "Continuous Monitoring",
        "description": "IMU always active, frequent sensor polling",
        "sensor_config": {
            "imu_mode": "typical",
            "polling_frequency": 60  # times per hour
        }
    },
    "PERIODIC": {
        "name": "Periodic Monitoring",
        "description": "IMU always active, reduced sensor polling",
        "sensor_config": {
            "imu_mode": "low_power",
            "polling_frequency": 12  # times per hour
        }
    },
    "POWER_SAVE": {
        "name": "Power Save",
        "description": "IMU in low power, minimal sensor polling",
        "sensor_config": {
            "imu_mode": "low_power",
            "polling_frequency": 4   # times per hour
        }
    }
}

# =============================================================================
# CALCULATION FUNCTIONS
# =============================================================================

def calculate_sensor_consumption(sensors_config, hours=24):
    """
    Calculate power consumption for sensors with different operation modes.
    
    Args:
        sensors_config: Dictionary of sensor configurations
        hours: Time period to calculate for (default: 24 hours)
    
    Returns:
        Dictionary containing power consumption details and total consumption in mWh
    """
    consumption = {
        "continuous": 0.0,
        "polled": 0.0,
        "total": 0.0,
        "details": {}
    }
    
    base_frequency = sensors_config.get('base_frequency_per_hour', 60.0)
    base_duration = sensors_config.get('base_duration_seconds', 0.1)
    
    # Process each sensor
    for sensor_id, sensor in sensors_config.items():
        if sensor_id not in ['base_frequency_per_hour', 'base_duration_seconds']:
            if sensor['enabled']:
                sensor_type = SENSOR_CONFIGS[sensor_id]
                sensor_consumption = 0.0
                
                if sensor_type["operation_mode"] == "continuous":
                    # Continuous operation (e.g., IMU)
                    sensor_consumption = sensor['active_power'] * hours
                    consumption["continuous"] += sensor_consumption
                else:
                    # Polled operation
                    active_hours = (base_frequency * base_duration * hours) / 3600
                    sleep_hours = hours - active_hours
                    
                    sensor_consumption = (
                        sensor['active_power'] * active_hours +
                        sensor['sleep_power'] * sleep_hours
                    )
                    consumption["polled"] += sensor_consumption
                
                # Store individual sensor details
                consumption["details"][sensor_id] = {
                    "total_mwh": sensor_consumption,
                    "average_mw": sensor_consumption / hours,
                    "operation_mode": sensor_type["operation_mode"]
                }
    
    consumption["total"] = consumption["continuous"] + consumption["polled"]
    return consumption

def calculate_comms_consumption(comms_config, hours=24):
    """
    Calculate power consumption for communication devices.
    
    Args:
        comms_config: Dictionary of communication configurations
        hours: Time period to calculate for (default: 24 hours)
    
    Returns:
        Dictionary containing power consumption details and total consumption in mWh
    """
    consumption = {
        "total": 0.0,
        "details": {}
    }
    
    # GPS Consumption
    if comms_config['gps']['enabled']:
        gps = comms_config['gps']
        gps_consumption = 0.0
        
        # Calculate acquisition and tracking power
        updates_per_period = gps['frequency_per_hour'] * hours
        
        # First update includes acquisition
        acquisition_energy = (COMMS_CONFIGS['GPS']['power_modes']['acquisition']['power'] * 
                            COMMS_CONFIGS['GPS']['power_modes']['acquisition']['typical_duration'] / 3600)
        
        # Subsequent updates use tracking power
        tracking_energy = ((updates_per_period - 1) * gps['active_power'] * 
                         gps['duration_seconds'] / 3600)
        
        gps_consumption = acquisition_energy + tracking_energy
        
        consumption['details']['gps'] = {
            "total_mwh": gps_consumption,
            "average_mw": gps_consumption / hours,
            "updates": updates_per_period
        }
        consumption['total'] += gps_consumption
    
    # Cellular Consumption
    if comms_config['cellular']['enabled']:
        cellular = comms_config['cellular']
        cellular_consumption = 0.0
        
        # Calculate startup and active power
        sessions_per_period = cellular['frequency_per_day'] * (hours / 24)
        
        # Startup energy
        startup_energy = (sessions_per_period * 
                         COMMS_CONFIGS['CELLULAR']['power_modes']['startup']['power'] * 
                         COMMS_CONFIGS['CELLULAR']['power_modes']['startup']['duration'] / 3600)
        
        # Active transmission energy
        active_energy = (sessions_per_period * cellular['active_power'] * 
                        cellular['duration_minutes'] * 60 / 3600)
        
        cellular_consumption = startup_energy + active_energy
        
        consumption['details']['cellular'] = {
            "total_mwh": cellular_consumption,
            "average_mw": cellular_consumption / hours,
            "sessions": sessions_per_period
        }
        consumption['total'] += cellular_consumption
    
    # LoRa Consumption
    if comms_config['lora']['enabled']:
        lora = comms_config['lora']
        lora_consumption = 0.0
        
        # Calculate transmission power
        messages_per_period = (hours if lora['frequency_type'] == 'per_hour' 
                             else (hours/24)) * lora['frequency']
        
        tx_energy = (messages_per_period * lora['active_power'] * 
                    lora['duration_seconds'] / 3600)
        
        # Calculate receive power if listening is enabled
        rx_energy = 0
        if lora.get('listen_enabled', False):
            rx_time = hours * 3600 * lora.get('rx_duty_cycle', 0.1)  # seconds
            rx_energy = (COMMS_CONFIGS['LORA']['power_modes']['rx']['power'] * 
                        rx_time / 3600)
        
        # Sleep power for remaining time
        active_time = (messages_per_period * lora['duration_seconds'] + 
                      (rx_time if lora.get('listen_enabled', False) else 0))
        sleep_time = hours * 3600 - active_time
        sleep_energy = (COMMS_CONFIGS['LORA']['power_modes']['sleep']['power'] * 
                       sleep_time / 3600)
        
        lora_consumption = tx_energy + rx_energy + sleep_energy
        
        consumption['details']['lora'] = {
            "total_mwh": lora_consumption,
            "average_mw": lora_consumption / hours,
            "messages": messages_per_period,
            "rx_enabled": lora.get('listen_enabled', False)
        }
        consumption['total'] += lora_consumption
    
    return consumption

def calculate_coprocessor_consumption(config, hours=24):
    """
    Calculate power consumption for coprocessor including startup costs.
    
    Args:
        config: Coprocessor configuration dictionary
        hours: Time period to calculate for (default: 24 hours)
    
    Returns:
        Dictionary containing power consumption details and total consumption in mWh
    """
    if not config['enabled']:
        return {"total": 0.0, "details": {}}
    
    consumption = {"details": {}}
    
    # Calculate number of processing windows
    windows_per_period = config['frequency_per_day'] * (hours / 24)
    
    # Calculate startup energy
    startup_time_hrs = COPROCESSOR_CONFIGS[config['type']]['startup_time'] / 3600
    startup_power = COPROCESSOR_CONFIGS[config['type']]['power_modes']['typical']['active']
    startup_energy = startup_power * startup_time_hrs * windows_per_period
    
    # Calculate active processing energy
    active_time_hrs = (config['duration_minutes'] * windows_per_period) / 60
    active_energy = config['active_power'] * active_time_hrs
    
    # Calculate idle energy
    idle_time_hrs = hours - active_time_hrs - (startup_time_hrs * windows_per_period)
    idle_energy = config['idle_power'] * idle_time_hrs
    
    total_energy = startup_energy + active_energy + idle_energy
    
    consumption['details'] = {
        "startup_mwh": startup_energy,
        "active_mwh": active_energy,
        "idle_mwh": idle_energy,
        "average_mw": total_energy / hours,
        "windows": windows_per_period
    }
    consumption['total'] = total_energy
    
    return consumption

def calculate_total_consumption(sensors_config, comms_config, coprocessor_config, 
                              efficiency_factors, hours=24):
    """
    Calculate total system power consumption with efficiency factors.
    
    Args:
        sensors_config: Sensor configuration dictionary
        comms_config: Communications configuration dictionary
        coprocessor_config: Coprocessor configuration dictionary
        efficiency_factors: Dictionary of efficiency factors (temperature, aging, voltage)
        hours: Time period to calculate for (default: 24 hours)
    
    Returns:
        Dictionary containing detailed power consumption breakdown and totals
    """
    consumption = {
        "sensors": calculate_sensor_consumption(sensors_config, hours),
        "communications": calculate_comms_consumption(comms_config, hours),
        "coprocessor": calculate_coprocessor_consumption(coprocessor_config, hours),
        "efficiency_factors": efficiency_factors
    }
    
    # Calculate raw total
    raw_total = (consumption["sensors"]["total"] + 
                 consumption["communications"]["total"] + 
                 consumption["coprocessor"]["total"])
    
    # Apply efficiency factors
    efficiency = (efficiency_factors["temperature"] / 100 * 
                 efficiency_factors["aging"] / 100 * 
                 efficiency_factors["voltage"] / 100)
    
    consumption["raw_total_mwh"] = raw_total
    consumption["efficiency"] = efficiency
    consumption["derated_total_mwh"] = raw_total / efficiency
    consumption["average_power_mw"] = consumption["derated_total_mwh"] / hours
    
    return consumption

def calculate_battery_life(battery_capacity_wh, daily_consumption_mwh):
    """Calculate expected battery life in days."""
    battery_capacity_mwh = battery_capacity_wh * 1000
    return battery_capacity_mwh / daily_consumption_mwh if daily_consumption_mwh > 0 else float('inf')

# =============================================================================
# STREAMLIT UI COMPONENTS
# =============================================================================

def create_system_mode_section():
    """Create the system operation mode selection section."""
    st.header("System Operation Mode")
    st.markdown("---")
    
    mode = st.selectbox(
        "Select System Operation Mode:",
        list(SYSTEM_MODES.keys()),
        format_func=lambda x: SYSTEM_MODES[x]["name"]
    )
    
    st.info(SYSTEM_MODES[mode]["description"])
    return mode

def create_sensor_section(system_mode):
    """Create the sensor configuration section."""
    st.header("Sensor Configuration")
    st.markdown("---")
    
    # Get mode-specific sensor defaults
    mode_config = SYSTEM_MODES[system_mode]["sensor_config"]
    
    sensors_config = {}
    
    # Create columns for better organization
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Sensor Selection")
        
        # IMU is always enabled but mode can change
        st.write("**IMU Configuration (Always On)**")
        imu_mode = mode_config["imu_mode"]
        sensors_config["LSM6DSV"] = {
            "enabled": True,
            "active_power": SENSOR_CONFIGS["LSM6DSV"][imu_mode]["active"],
            "sleep_power": SENSOR_CONFIGS["LSM6DSV"][imu_mode]["sleep"]
        }
        st.write(f"Mode: {imu_mode.title()}")
        st.write(f"Sampling Rate: {SENSOR_CONFIGS['LSM6DSV']['sampling_rates'][imu_mode]} Hz")
    
    with col2:
        st.subheader("Polled Sensor Configuration")
        
        # Configure other sensors
        for sensor_id in ["MMC5983MA", "TSL2591", "BME280"]:
            sensor = SENSOR_CONFIGS[sensor_id]
            with st.expander(f"{sensor['name']} Configuration"):
                enabled = st.checkbox("Enable Sensor", key=f"enable_{sensor_id}")
                if enabled:
                    power_mode = st.radio(
                        "Power Mode",
                        ["typical", "low_power"],
                        key=f"mode_{sensor_id}"
                    )
                    
                    sensors_config[sensor_id] = {
                        "enabled": True,
                        "active_power": sensor[power_mode]["active"],
                        "sleep_power": sensor[power_mode]["sleep"]
                    }
                else:
                    sensors_config[sensor_id] = {"enabled": False}
    
    # Global polling configuration
    st.subheader("Polling Configuration")
    # Convert mode_config polling_frequency to float
    default_frequency = float(mode_config["polling_frequency"])
    sensors_config['base_frequency_per_hour'] = st.number_input(
        "Base Sampling Frequency (times per hour):",
        value=default_frequency,
        min_value=1.0,
        step=1.0,  # Add step for better control
        help="How often to poll enabled sensors"
    )
    
    sensors_config['base_duration_seconds'] = st.number_input(
        "Sampling Duration (seconds):",
        value=0.1,
        min_value=0.1,
        step=0.1,  # Add step for better control
        format="%.1f",  # Format to one decimal place
        help="How long each sensor sampling takes"
    )
    
    return sensors_config

def create_communications_section():
    """Create the communications configuration section."""
    st.header("Communications Configuration")
    st.markdown("---")
    
    comms_config = {"gps": {}, "cellular": {}, "lora": {}}
    
    # GPS Configuration
    with st.expander("GPS Configuration (MAX-M10S)", expanded=True):
        gps_enabled = st.checkbox("Enable GPS", value=True)
        if gps_enabled:
            col1, col2 = st.columns(2)
            with col1:
                freq = st.number_input(
                    "Update Frequency (times per hour):",
                    value=COMMS_CONFIGS["GPS"]["default_schedule"]["frequency"],
                    min_value=0.1,
                    help="How often to get position fixes"
                )
            with col2:
                duration = st.number_input(
                    "Fix Duration (seconds):",
                    value=COMMS_CONFIGS["GPS"]["default_schedule"]["duration"],
                    min_value=1.0,
                    help="Time needed for each position fix"
                )
            
            st.info(f"Average power during fix: {COMMS_CONFIGS['GPS']['power_modes']['tracking']['power']} mW")
            comms_config["gps"] = {
                "enabled": True,
                "active_power": COMMS_CONFIGS["GPS"]["power_modes"]["tracking"]["power"],
                "frequency_per_hour": freq,
                "duration_seconds": duration
            }
        else:
            comms_config["gps"] = {"enabled": False}
    
    # Cellular Configuration
    with st.expander("Cellular Configuration (EG25-G)", expanded=True):
        cellular_enabled = st.checkbox("Enable Cellular", value=True)
        if cellular_enabled:
            col1, col2 = st.columns(2)
            with col1:
                freq = st.number_input(
                    "Updates per Day:",
                    value=COMMS_CONFIGS["CELLULAR"]["default_schedule"]["frequency"],
                    min_value=1,
                    help="Number of communication windows per day"
                )
            with col2:
                duration = st.number_input(
                    "Window Duration (minutes):",
                    value=COMMS_CONFIGS["CELLULAR"]["default_schedule"]["duration"],
                    min_value=1,
                    help="Duration of each communication window"
                )
            
            st.info(f"Power during transmission: {COMMS_CONFIGS['CELLULAR']['power_modes']['active']['power']} mW")
            comms_config["cellular"] = {
                "enabled": True,
                "active_power": COMMS_CONFIGS["CELLULAR"]["power_modes"]["active"]["power"],
                "frequency_per_day": freq,
                "duration_minutes": duration
            }
        else:
            comms_config["cellular"] = {"enabled": False}
    
    # LoRa Configuration
    with st.expander("LoRa Configuration (LLCC68)", expanded=True):
        lora_enabled = st.checkbox("Enable LoRa", value=True)
        if lora_enabled:
            col1, col2 = st.columns(2)
            with col1:
                freq_type = st.selectbox(
                    "Message Frequency Type:",
                    ["per_hour", "per_day"]
                )
                freq = st.number_input(
                    f"Messages {freq_type}:",
                    value=1.0,
                    min_value=0.1
                )
            with col2:
                duration = st.number_input(
                    "Message Duration (seconds):",
                    value=COMMS_CONFIGS["LORA"]["power_modes"]["tx"]["typical_duration"],
                    min_value=0.1
                )
                listen_enabled = st.checkbox("Enable Receive Mode", value=False)
            
            if listen_enabled:
                rx_duty_cycle = st.slider(
                    "Receive Duty Cycle (%):",
                    min_value=1,
                    max_value=100,
                    value=10,
                    help="Percentage of time spent listening for messages"
                ) / 100.0
            
            comms_config["lora"] = {
                "enabled": True,
                "active_power": COMMS_CONFIGS["LORA"]["power_modes"]["tx"]["power"],
                "frequency_type": freq_type,
                "frequency": freq,
                "duration_seconds": duration,
                "listen_enabled": listen_enabled,
                "rx_duty_cycle": rx_duty_cycle if listen_enabled else 0
            }
        else:
            comms_config["lora"] = {"enabled": False}
    
    return comms_config

def create_coprocessor_section():
    """Create the coprocessor configuration section."""
    st.header("Co-Processor Configuration")
    st.markdown("---")
    
    coprocessor_enabled = st.checkbox("Enable Co-Processor", value=False)
    if coprocessor_enabled:
        col1, col2 = st.columns(2)
        
        with col1:
            coprocessor_type = st.selectbox(
                "Select Co-Processor:",
                list(COPROCESSOR_CONFIGS.keys())
            )
            power_mode = st.radio(
                "Power Mode:",
                ["typical", "max"]
            )
            
        with col2:
            freq = st.number_input(
                "Processing Windows per Day:",
                value=1,
                min_value=1,
                help="Number of processing sessions per day"
            )
            duration = st.number_input(
                "Window Duration (minutes):",
                value=5,
                min_value=1,
                help="Duration of each processing window"
            )
        
        config = {
            "enabled": True,
            "type": coprocessor_type,
            "active_power": COPROCESSOR_CONFIGS[coprocessor_type]["power_modes"][power_mode]["active"],
            "idle_power": COPROCESSOR_CONFIGS[coprocessor_type]["power_modes"][power_mode]["idle"],
            "frequency_per_day": freq,
            "duration_minutes": duration
        }
        
        # Show power usage warning if high
        if config["active_power"] * duration/60 > 1000:  # If over 1Wh per window
            st.warning("⚠️ High power consumption detected. Consider reducing processing window duration.")
    else:
        config = {"enabled": False}
    
    return config

def create_battery_section():
    """Create the battery configuration section."""
    st.header("Battery Configuration")
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Battery Selection")
        battery_type = st.radio(
            "Select Battery:",
            list(BATTERY_CONFIGS.keys()),
            format_func=lambda x: f"{BATTERY_CONFIGS[x]['capacity']}Wh Battery"
        )
    
    with col2:
        st.subheader("Derating Factors")
        temperature_derating = st.slider(
            "Temperature Derating (%)", 
            min_value=70, 
            max_value=100, 
            value=BATTERY_CONFIGS[battery_type]["typical_derating"]["temperature"]
        )
        
        aging_factor = st.slider(
            "Aging Factor (%)", 
            min_value=70, 
            max_value=100, 
            value=BATTERY_CONFIGS[battery_type]["typical_derating"]["aging"]
        )
        
        voltage_efficiency = st.slider(
            "Voltage Efficiency (%)",
            min_value=80,
            max_value=95,
            value=BATTERY_CONFIGS[battery_type]["typical_derating"]["voltage"]
        )
    
    return {
        "type": battery_type,
        "capacity": BATTERY_CONFIGS[battery_type]["capacity"],
        "derating": {
            "temperature": temperature_derating,
            "aging": aging_factor,
            "voltage": voltage_efficiency
        }
    }

def calculate_power_timeline(consumption_data, hours=24):
    """
    Calculate detailed power timeline with all components.
    Returns power values for each second of the day.
    """
    # Initialize timeline with one second resolution
    seconds = hours * 3600
    timeline = np.zeros(seconds)
    
    # 1. Add continuous sensor power (e.g., IMU)
    if consumption_data["sensors"]["continuous"] > 0:
        continuous_power = consumption_data["sensors"]["continuous"] / hours
        timeline += continuous_power
    
    # 2. Add polled sensor events
    if "details" in consumption_data["sensors"]:
        for sensor_id, sensor in consumption_data["sensors"]["details"].items():
            if sensor["operation_mode"] == "polled":
                # Calculate timing
                interval = 3600 / consumption_data.get("base_frequency_per_hour", 60)
                duration = consumption_data.get("base_duration_seconds", 0.1)
                
                # Add sensor active periods
                for t in range(0, seconds, int(interval)):
                    if t + duration < seconds:
                        timeline[int(t):int(t + duration)] += (sensor["average_mw"])
    
    # 3. Add GPS events
    comms_details = consumption_data.get("communications", {}).get("details", {})
    if "gps" in comms_details:
        gps_details = comms_details["gps"]
        if isinstance(gps_details, dict) and gps_details.get("average_mw"):
            interval = int(3600 * 24 / gps_details.get("updates", 24))  # Fallback to hourly if no updates specified
            duration = int(COMMS_CONFIGS["GPS"]["power_modes"]["tracking"]["typical_duration"])
            
            for t in range(0, seconds, interval):
                if t == 0:  # First fix includes acquisition
                    timeline[t:t + int(COMMS_CONFIGS["GPS"]["power_modes"]["acquisition"]["typical_duration"])] += \
                        COMMS_CONFIGS["GPS"]["power_modes"]["acquisition"]["power"]
                else:  # Regular tracking
                    if t + duration < seconds:
                        timeline[int(t):int(t + duration)] += COMMS_CONFIGS["GPS"]["power_modes"]["tracking"]["power"]
    
    # 4. Add Cellular events
    if "cellular" in comms_details:
        cellular_details = comms_details["cellular"]
        if isinstance(cellular_details, dict) and cellular_details.get("average_mw"):
            # Calculate sessions based on total updates
            sessions = cellular_details.get("sessions", 1)
            interval = int(24 * 3600 / sessions)
            duration = int(COMMS_CONFIGS["CELLULAR"]["default_schedule"]["duration"] * 60)  # Convert minutes to seconds
            
            for t in range(0, seconds, interval):
                if t + duration < seconds:
                    # Add startup spike
                    timeline[t] += COMMS_CONFIGS["CELLULAR"]["power_modes"]["startup"]["power"]
                    # Add transmission window
                    timeline[t+1:t+duration] += COMMS_CONFIGS["CELLULAR"]["power_modes"]["active"]["power"]
    
    # 5. Add LoRa events
    if "lora" in comms_details:
        lora_details = comms_details["lora"]
        if isinstance(lora_details, dict) and lora_details.get("average_mw"):
            messages = lora_details.get("messages", 24)  # Default to hourly
            interval = int(24 * 3600 / messages)
            duration = int(COMMS_CONFIGS["LORA"]["power_modes"]["tx"]["typical_duration"])
            
            for t in range(0, seconds, interval):
                if t + duration < seconds:
                    timeline[t:t+duration] += COMMS_CONFIGS["LORA"]["power_modes"]["tx"]["power"]
            
            # Add listening power if enabled
            if lora_details.get("rx_enabled", False):
                rx_power = COMMS_CONFIGS["LORA"]["power_modes"]["rx"]["power"]
                timeline += rx_power * lora_details.get("rx_duty_cycle", 0.1)
    
    # 6. Add Coprocessor events
    if consumption_data["coprocessor"].get("total", 0) > 0:
        coprocessor_details = consumption_data["coprocessor"].get("details", {})
        if isinstance(coprocessor_details, dict):
            windows = coprocessor_details.get("windows", 1)
            interval = int(24 * 3600 / windows)
            duration = int(coprocessor_details.get("active_mwh", 0) / coprocessor_details.get("average_mw", 1) * 3600)
            
            for t in range(0, seconds, interval):
                if t + duration < seconds:
                    timeline[t:t+duration] += coprocessor_details.get("average_mw", 0)
    
    return timeline

def plot_battery_life(daily_consumption_mwh, target_batteries=[77.0, 100.7]):
    """Create enhanced battery life visualization with accurate power profile."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12), height_ratios=[2, 1])
    
    # Plot 1: Battery Life Curve
    battery_capacities = np.linspace(10, 150, 100)
    battery_lives = [calculate_battery_life(cap, daily_consumption_mwh) 
                    for cap in battery_capacities]
    
    ax1.plot(battery_capacities, battery_lives, 'b-', label='Battery Life')
    
    # Add target battery vertical lines and annotations
    for capacity in target_batteries:
        life = calculate_battery_life(capacity, daily_consumption_mwh)
        ax1.axvline(x=capacity, color='r', linestyle='--')
        ax1.axhline(y=life, color='g', linestyle=':', alpha=0.5)
        
        # Add annotation
        ax1.annotate(
            f'{capacity}Wh: {life:.1f} days',
            xy=(capacity, life),
            xytext=(10, 10),
            textcoords='offset points',
            ha='left',
            va='bottom',
            bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0')
        )
    
    # Add lifetime zones with better visibility
    ax1.axhspan(0, 30, alpha=0.1, color='red', label='< 30 days')
    ax1.axhspan(30, 90, alpha=0.1, color='yellow', label='30-90 days')
    ax1.axhspan(90, max(battery_lives), alpha=0.1, color='green', label='> 90 days')
    
    ax1.set_xlabel('Battery Capacity (Wh)')
    ax1.set_ylabel('Battery Life (Days)')
    ax1.set_title('Battery Life vs. Capacity')
    ax1.grid(True)
    ax1.legend()

    # Plot 2: Daily Power Profile - Using the improved timeline calculation
    try:
        timeline = calculate_power_timeline(st.session_state.last_calculation)
        
        # Convert timeline to hourly averages for clearer visualization
        hours = np.arange(24)
        hourly_power = np.array([np.mean(timeline[i*3600:(i+1)*3600]) for i in range(24)])
        
        # Plot the hourly power
        ax2.plot(hours, hourly_power, 'b-', label='Power Profile')
        
        # Add average line
        avg_power = np.mean(hourly_power)
        ax2.axhline(y=avg_power, color='r', linestyle='--', 
                   label=f'Average ({avg_power:.1f} mW)')
        
        # Add min/max annotations
        max_power = np.max(hourly_power)
        min_power = np.min(hourly_power)
        ax2.annotate(f'Peak: {max_power:.1f} mW', 
                    xy=(hours[np.argmax(hourly_power)], max_power),
                    xytext=(5, 5), textcoords='offset points')
        ax2.annotate(f'Min: {min_power:.1f} mW',
                    xy=(hours[np.argmin(hourly_power)], min_power),
                    xytext=(5, -15), textcoords='offset points')
        
    except Exception as e:
        # Fallback to simple average power display if timeline calculation fails
        hours = np.arange(24)
        avg_power = st.session_state.last_calculation["average_power_mw"]
        ax2.plot(hours, [avg_power] * 24, 'b-', label='Average Power')
    
    ax2.set_xlabel('Hour of Day')
    ax2.set_ylabel('Power Draw (mW)')
    ax2.set_title('24-Hour Power Profile')
    ax2.grid(True)
    ax2.legend()

    plt.tight_layout()
    return fig

def plot_power_distribution(consumption_data):
    """Create detailed power distribution visualization."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Pie chart of average power distribution
    components = []
    
    # Add sensor components
    if consumption_data["sensors"]["continuous"] > 0:
        components.append(("Continuous Sensors", 
                         consumption_data["sensors"]["continuous"]/24))
    if consumption_data["sensors"]["polled"] > 0:
        components.append(("Polled Sensors", 
                         consumption_data["sensors"]["polled"]/24))
    
    # Add communication components
    for comm_type, details in consumption_data["communications"]["details"].items():
        components.append((comm_type.upper(), details["average_mw"]))
    
    # Add coprocessor if enabled
    if consumption_data["coprocessor"]["total"] > 0:
        components.append(("Co-Processor", 
                         consumption_data["coprocessor"]["total"]/24))
    
    # Create pie chart
    labels = [c[0] for c in components]
    sizes = [c[1] for c in components]
    
    ax1.pie(sizes, labels=labels, autopct='%1.1f%%', 
            startangle=90, counterclock=False)
    ax1.axis('equal')
    ax1.set_title('Average Power Distribution')

    # Stacked bar chart of daily energy consumption
    categories = {
        'Sensors': consumption_data["sensors"]["total"],
        'Communications': consumption_data["communications"]["total"],
        'Co-Processor': consumption_data["coprocessor"]["total"]
    }
    
    bottom = 0
    bars = []
    for category, value in categories.items():
        if value > 0:
            bar = ax2.bar('Daily Energy', value, bottom=bottom, label=category)
            bottom += value
            bars.append(bar)
    
    ax2.set_ylabel('Energy Consumption (mWh)')
    ax2.set_title('Daily Energy Breakdown')
    ax2.legend()

    # Add total values
    ax2.text(0, bottom + 50, f'Total: {consumption_data["raw_total_mwh"]:.1f} mWh\n' +
             f'Derated: {consumption_data["derated_total_mwh"]:.1f} mWh',
             ha='center', va='bottom')

    plt.tight_layout()
    return fig

def display_results(consumption_data, battery_config):
    """Display comprehensive results and analysis."""
    st.header("Power Analysis Results")
    st.markdown("---")
    
    # Summary metrics in columns
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="Average Power Draw",
            value=f"{consumption_data['average_power_mw']:.2f} mW",
            help="Average power consumption including all factors"
        )
    
    with col2:
        st.metric(
            label="Daily Energy Consumption",
            value=f"{consumption_data['derated_total_mwh']:.2f} mWh",
            help="Total daily energy consumption after derating"
        )
    
    with col3:
        efficiency = consumption_data["efficiency"] * 100
        st.metric(
            label="System Efficiency",
            value=f"{efficiency:.1f}%",
            help="Combined effect of all efficiency factors"
        )
    
    # Detailed breakdowns in tabs
    tab1, tab2, tab3 = st.tabs(["Component Analysis", "Battery Life", "Power Profile"])
    
    with tab1:
        st.subheader("Power Distribution")
        fig_dist = plot_power_distribution(consumption_data)
        st.pyplot(fig_dist)
        
        # Detailed component breakdown
        st.subheader("Component Details")
        
        with st.expander("Sensor Power", expanded=True):
            st.write("Continuous Sensors:")
            st.write(f"- Average Power: {consumption_data['sensors']['continuous']/24:.2f} mW")
            st.write("Polled Sensors:")
            st.write(f"- Average Power: {consumption_data['sensors']['polled']/24:.2f} mW")
            for sensor_id, details in consumption_data["sensors"]["details"].items():
                st.write(f"{sensor_id}:")
                st.write(f"- Average Power: {details['average_mw']:.2f} mW")
        
        with st.expander("Communications Power", expanded=True):
            for comm_type, details in consumption_data["communications"]["details"].items():
                st.write(f"{comm_type.upper()}:")
                st.write(f"- Average Power: {details['average_mw']:.2f} mW")
                if 'updates' in details:
                    st.write(f"- Updates: {details['updates']:.1f} per day")
        
        if consumption_data["coprocessor"]["total"] > 0:
            with st.expander("Co-Processor Power", expanded=True):
                details = consumption_data["coprocessor"]["details"]
                st.write(f"Average Power: {details['average_mw']:.2f} mW")
                st.write(f"Processing Windows: {details['windows']:.1f} per day")
    
    with tab2:
        st.subheader("Battery Life Analysis")
        fig_life = plot_battery_life(consumption_data["derated_total_mwh"])
        st.pyplot(fig_life)
        
        # Battery life estimates
        st.subheader("Expected Battery Life")
        for capacity in [77.0, 100.7]:
            battery_life = calculate_battery_life(capacity, 
                                               consumption_data["derated_total_mwh"])
            months = battery_life / 30.44
            
            if battery_life < 30:
                st.error(f"{capacity}Wh Battery: {battery_life:.1f} days ({months:.1f} months)")
            elif battery_life < 90:
                st.warning(f"{capacity}Wh Battery: {battery_life:.1f} days ({months:.1f} months)")
            else:
                st.success(f"{capacity}Wh Battery: {battery_life:.1f} days ({months:.1f} months)")
    
    with tab3:
        st.subheader("24-Hour Power Profile")
        
        try:
            # Calculate detailed timeline
            timeline = calculate_power_timeline(consumption_data)
            
            # Convert to hourly averages
            hours = np.arange(24)
            hourly_power = np.array([np.mean(timeline[i*3600:(i+1)*3600]) for i in range(24)])
            
            col1, col2 = st.columns([3, 1])
            
            with col1:
                # Create main power profile plot
                fig_profile = plt.figure(figsize=(12, 8))
                plt.subplot(2, 1, 1)
                
                # Plot hourly averages as bars
                plt.bar(hours, hourly_power, alpha=0.5, color='skyblue', label='Hourly Average')
                plt.axhline(y=consumption_data["average_power_mw"], 
                          color='r', linestyle='--', 
                          label=f'Daily Average ({consumption_data["average_power_mw"]:.1f} mW)')
                
                plt.xlabel('Hour of Day')
                plt.ylabel('Power Draw (mW)')
                plt.title('Hourly Average Power Profile')
                plt.grid(True, alpha=0.3)
                plt.legend()
                
                # Add detailed timeline subplot
                plt.subplot(2, 1, 2)
                detailed_minutes = np.arange(24*60) / 60  # X-axis in hours
                minute_averages = np.array([np.mean(timeline[i*60:(i+1)*60]) 
                                          for i in range(24*60)])
                
                plt.plot(detailed_minutes, minute_averages, 
                        linewidth=1, alpha=0.7, color='blue',
                        label='Minute-by-Minute')
                plt.axhline(y=consumption_data["average_power_mw"], 
                          color='r', linestyle='--', 
                          label='Daily Average')
                
                plt.xlabel('Hour of Day')
                plt.ylabel('Power Draw (mW)')
                plt.title('Detailed Power Profile (1-Minute Resolution)')
                plt.grid(True, alpha=0.3)
                plt.legend()
                
                plt.tight_layout()
                st.pyplot(fig_profile)
            
            with col2:
                st.write("Power Statistics:")
                st.write("---")
                
                # Calculate statistics
                peak_power = np.max(minute_averages)
                min_power = np.min(minute_averages)
                avg_power = np.mean(minute_averages)
                
                st.write(f"Peak Power: {peak_power:.2f} mW")
                st.write(f"Minimum Power: {min_power:.2f} mW")
                st.write(f"Average Power: {avg_power:.2f} mW")
                st.write("---")
                
                # Add active times
                st.write("Active Periods:")
                if consumption_data["communications"]["details"].get("cellular"):
                    cellular = consumption_data["communications"]["details"]["cellular"]
                    st.write(f"Cellular: {cellular.get('sessions', 0):.1f} times per day")
                
                if consumption_data["communications"]["details"].get("gps"):
                    gps = consumption_data["communications"]["details"]["gps"]
                    st.write(f"GPS: {gps.get('updates', 0):.1f} updates per day")
                
                if consumption_data["communications"]["details"].get("lora"):
                    lora = consumption_data["communications"]["details"]["lora"]
                    st.write(f"LoRa: {lora.get('messages', 0):.1f} messages per day")
                
                if consumption_data["coprocessor"]["total"] > 0:
                    coprocessor = consumption_data["coprocessor"]["details"]
                    st.write(f"Co-Processor: {coprocessor.get('windows', 0):.1f} windows per day")
                
                st.write("---")
                
                # Add power distribution
                st.write("Power Contribution:")
                total_power = consumption_data["average_power_mw"]
                if total_power > 0:
                    if consumption_data["sensors"]["continuous"] > 0:
                        sensor_cont_pct = (consumption_data["sensors"]["continuous"]/24) / total_power * 100
                        st.write(f"Continuous Sensors: {sensor_cont_pct:.1f}%")
                    
                    if consumption_data["sensors"]["polled"] > 0:
                        sensor_poll_pct = (consumption_data["sensors"]["polled"]/24) / total_power * 100
                        st.write(f"Polled Sensors: {sensor_poll_pct:.1f}%")
                    
                    if consumption_data["communications"]["total"] > 0:
                        comms_pct = (consumption_data["communications"]["total"]/24) / total_power * 100
                        st.write(f"Communications: {comms_pct:.1f}%")
                    
                    if consumption_data["coprocessor"]["total"] > 0:
                        coproc_pct = (consumption_data["coprocessor"]["total"]/24) / total_power * 100
                        st.write(f"Co-Processor: {coproc_pct:.1f}%")
        
        except Exception as e:
            st.error("Unable to generate detailed power profile. Using average power only.")
            
            # Fallback to simple average display
            fig_profile = plt.figure(figsize=(10, 6))
            hours = np.arange(24)
            power_profile = [consumption_data["average_power_mw"]] * 24
            plt.plot(hours, power_profile, 'b-', label='Average Power')
            plt.xlabel('Hour of Day')
            plt.ylabel('Power Draw (mW)')
            plt.title('24-Hour Average Power Profile')
            plt.grid(True)
            plt.legend()
            st.pyplot(fig_profile)
            
            with col2:
                st.write("Power Statistics:")
                st.write("---")
                st.write(f"Average Power: {consumption_data['average_power_mw']:.2f} mW")

    # Power optimization recommendations
    st.header("Optimization Recommendations")
    st.markdown("---")
    
    # Generate recommendations based on power analysis
    if consumption_data["average_power_mw"] > 100:
        st.warning("High average power consumption detected")
        
        # Find highest power consumer
        components = []
        if consumption_data["sensors"]["total"] > 0:
            components.append(("Sensors", consumption_data["sensors"]["total"]))
        if consumption_data["communications"]["total"] > 0:
            components.append(("Communications", consumption_data["communications"]["total"]))
        if consumption_data["coprocessor"]["total"] > 0:
            components.append(("Co-Processor", consumption_data["coprocessor"]["total"]))
        
        max_component = max(components, key=lambda x: x[1])
        st.write(f"Highest power consumer: {max_component[0]}")
        
        if max_component[0] == "Communications":
            st.write("Consider:")
            st.write("- Reducing communication frequency")
            st.write("- Optimizing data transmission size")
            st.write("- Using more efficient communication modes")
        elif max_component[0] == "Co-Processor":
            st.write("Consider:")
            st.write("- Reducing processing window duration")
            st.write("- Using lower power modes when possible")
            st.write("- Optimizing processing algorithms")
        elif max_component[0] == "Sensors":
            st.write("Consider:")
            st.write("- Using low power modes where possible")
            st.write("- Optimizing polling frequency")
            st.write("- Implementing smarter wake-up conditions")

# Add this to your main Streamlit app:
def main():
    st.set_page_config(page_title="Battery Life Calculator", layout="wide")
    
    st.title("Battery Life Calculator")
    
    # Create configuration sections
    system_mode = create_system_mode_section()
    sensors_config = create_sensor_section(system_mode)
    comms_config = create_communications_section()
    coprocessor_config = create_coprocessor_section()
    battery_config = create_battery_section()
    
    if st.button("Calculate Battery Life"):
        # Calculate consumption
        consumption_data = calculate_total_consumption(
            sensors_config,
            comms_config,
            coprocessor_config,
            battery_config["derating"]
        )
        
        # Store in session state
        st.session_state.last_calculation = consumption_data
        
        # Display results
        display_results(consumption_data, battery_config)

if __name__ == "__main__":
    main()