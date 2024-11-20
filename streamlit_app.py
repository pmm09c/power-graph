import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# =============================================================================
# CONSTANTS AND DEFAULT VALUES
# =============================================================================
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
SENSOR_CONFIGS = {
    "LSM6DSV": {
        "name": "LSM6DSV (IMU)",
        "description": "Always-on motion sensing",
        "operation_mode": "continuous",
        "typical": {
            "active": 0.695,  # mW
            "sleep": 0.0015   # mW
        },
        "low_power": {
            "active": 0.29,   # mW
            "sleep": 0.0015   # mW
        },
        "sampling_rates": {
            "typical": 104,   # Hz
            "low_power": 52   # Hz
        }
    },
    "MMC5983MA": {
        "name": "MMC5983MA (Magnetometer)",
        "description": "Magnetic field sensing",
        "operation_mode": "polled",
        "typical": {
            "active": 1.0,
            "sleep": 0.002
        },
        "low_power": {
            "active": 0.5,
            "sleep": 0.002
        },
        "min_sampling_period": 0.1
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
        "min_sampling_period": 0.1
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
        "min_sampling_period": 0.1
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

DEFAULT_PROFILES = {
    "Dumb Tracker": {
        "description": "Basic tracker with minimal processing - LTE, GPS and environmental sensors polling hourly",
        "sensor_config": {
            "LSM6DSV": {"enabled": False},
            "MMC5983MA": {"enabled": False},
            "TSL2591": {"enabled": False},
            "BME280": {
                "enabled": True,
                "active_power": SENSOR_CONFIGS["BME280"]["low_power"]["active"],
                "sleep_power": SENSOR_CONFIGS["BME280"]["low_power"]["sleep"]
            },
            "base_frequency_per_hour": 1.0,
            "base_duration_seconds": 0.1
        },
        "comms_config": {
            "gps": {
                "enabled": True,
                "active_power": COMMS_CONFIGS["GPS"]["power_modes"]["tracking"]["power"],
                "frequency_per_hour": 1.0,
                "duration_seconds": 30.0
            },
            "cellular": {
                "enabled": True,
                "active_power": COMMS_CONFIGS["CELLULAR"]["power_modes"]["active"]["power"],
                "frequency_per_day": 24,
                "duration_minutes": 1
            },
            "lora": {"enabled": False}
        },
        "coprocessor_config": {"enabled": False}
    },
    "Phase I Tracker": {
        "description": "High-power continuous monitoring with all sensors and co-processor active",
        "sensor_config": {
            "LSM6DSV": {
                "enabled": True,
                "active_power": SENSOR_CONFIGS["LSM6DSV"]["typical"]["active"],
                "sleep_power": SENSOR_CONFIGS["LSM6DSV"]["typical"]["sleep"]
            },
            "MMC5983MA": {
                "enabled": True,
                "active_power": SENSOR_CONFIGS["MMC5983MA"]["typical"]["active"],
                "sleep_power": SENSOR_CONFIGS["MMC5983MA"]["typical"]["sleep"]
            },
            "TSL2591": {
                "enabled": True,
                "active_power": SENSOR_CONFIGS["TSL2591"]["typical"]["active"],
                "sleep_power": SENSOR_CONFIGS["TSL2591"]["typical"]["sleep"]
            },
            "BME280": {
                "enabled": True,
                "active_power": SENSOR_CONFIGS["BME280"]["typical"]["active"],
                "sleep_power": SENSOR_CONFIGS["BME280"]["typical"]["sleep"]
            },
            "base_frequency_per_hour": 60.0,
            "base_duration_seconds": 0.1
        },
        "comms_config": {
            "gps": {
                "enabled": True,
                "active_power": COMMS_CONFIGS["GPS"]["power_modes"]["tracking"]["power"],
                "frequency_per_hour": 60.0,
                "duration_seconds": 30.0
            },
            "cellular": {
                "enabled": True,
                "active_power": COMMS_CONFIGS["CELLULAR"]["power_modes"]["active"]["power"],
                "frequency_per_day": 24,
                "duration_minutes": 5
            },
            "lora": {"enabled": False}
        },
        "coprocessor_config": {
            "enabled": True,
            "type": "Jetson Orin Nano",
            "active_power": COPROCESSOR_CONFIGS["Jetson Orin Nano"]["power_modes"]["typical"]["active"],
            "idle_power": COPROCESSOR_CONFIGS["Jetson Orin Nano"]["power_modes"]["typical"]["idle"],
            "frequency_per_day": 24,
            "duration_minutes": 60
        }
    },
    "Recommended": {
        "description": "Optimized power profile with balanced monitoring and processing using LoRa and CM4",
        "sensor_config": {
            "LSM6DSV": {
                "enabled": True,
                "active_power": SENSOR_CONFIGS["LSM6DSV"]["low_power"]["active"],
                "sleep_power": SENSOR_CONFIGS["LSM6DSV"]["low_power"]["sleep"]
            },
            "MMC5983MA": {
                "enabled": True,
                "active_power": SENSOR_CONFIGS["MMC5983MA"]["low_power"]["active"],
                "sleep_power": SENSOR_CONFIGS["MMC5983MA"]["low_power"]["sleep"]
            },
            "TSL2591": {
                "enabled": True,
                "active_power": SENSOR_CONFIGS["TSL2591"]["low_power"]["active"],
                "sleep_power": SENSOR_CONFIGS["TSL2591"]["low_power"]["sleep"]
            },
            "BME280": {
                "enabled": True,
                "active_power": SENSOR_CONFIGS["BME280"]["low_power"]["active"],
                "sleep_power": SENSOR_CONFIGS["BME280"]["low_power"]["sleep"]
            },
            "base_frequency_per_hour": 60.0,
            "base_duration_seconds": 0.1
        },
        "comms_config": {
            "gps": {
                "enabled": True,
                "active_power": COMMS_CONFIGS["GPS"]["power_modes"]["tracking"]["power"],
                "frequency_per_hour": 6.0,
                "duration_seconds": 30.0
            },
            "cellular": {
                "enabled": True,
                "active_power": COMMS_CONFIGS["CELLULAR"]["power_modes"]["active"]["power"],
                "frequency_per_day": 24,
                "duration_minutes": 1
            },
            "lora": {
                "enabled": True,
                "active_power": COMMS_CONFIGS["LORA"]["power_modes"]["tx"]["power"],
                "frequency_type": "per_hour",
                "frequency": 1.0,
                "duration_seconds": COMMS_CONFIGS["LORA"]["power_modes"]["tx"]["typical_duration"],
                "listen_enabled": True,
                "rx_duty_cycle": 0.1
            }
        },
        "coprocessor_config": {
            "enabled": True,
            "type": "Raspberry Pi CM4",
            "power_mode": "typical",  # Added explicit power mode selection
            "active_power": COPROCESSOR_CONFIGS["Raspberry Pi CM4"]["power_modes"]["typical"]["active"],
            "idle_power": COPROCESSOR_CONFIGS["Raspberry Pi CM4"]["power_modes"]["typical"]["idle"],
            "frequency_per_day": 4,
            "duration_minutes": 2
        }
    }
}

# =============================================================================
# CALCULATION FUNCTIONS
# =============================================================================

def calculate_sensor_consumption(sensors_config, hours=24):
    """Calculate power consumption for sensors."""
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
    """Calculate power consumption for communication devices."""
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
    """Calculate power consumption for coprocessor including startup costs."""
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
    """Calculate total system power consumption with efficiency factors."""
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

def calculate_power_timeline(consumption_data, hours=24):
    """Calculate detailed power timeline with all components."""
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
                        timeline[int(t):int(t + duration)] += sensor["average_mw"]
    
    # 3. Add GPS events
    comms_details = consumption_data.get("communications", {}).get("details", {})
    if "gps" in comms_details:
        gps_details = comms_details["gps"]
        if isinstance(gps_details, dict) and gps_details.get("average_mw"):
            interval = int(3600 * 24 / gps_details.get("updates", 24))
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
            sessions = cellular_details.get("sessions", 1)
            interval = int(24 * 3600 / sessions)
            duration = int(COMMS_CONFIGS["CELLULAR"]["default_schedule"]["duration"] * 60)
            
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
            messages = lora_details.get("messages", 24)
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
            duration = int(coprocessor_details.get("active_mwh", 0) / 
                         coprocessor_details.get("average_mw", 1) * 3600)
            
            for t in range(0, seconds, interval):
                if t + duration < seconds:
                    timeline[t:t+duration] += coprocessor_details.get("average_mw", 0)
    
    return timeline
# =============================================================================
# VISUALIZATION AND UI FUNCTIONS
# =============================================================================

def plot_battery_life(daily_consumption_mwh, target_batteries=[77.0, 100.7]):
    """Create enhanced battery life visualization."""
    fig, ax = plt.subplots(figsize=(10, 6))  # Single plot with adjusted size
    
    # Battery Life Curve
    battery_capacities = np.linspace(10, 150, 100)
    battery_lives = [calculate_battery_life(cap, daily_consumption_mwh) 
                    for cap in battery_capacities]
    
    ax.plot(battery_capacities, battery_lives, 'b-', label='Battery Life')
    
    # Add target battery vertical lines and annotations
    for capacity in target_batteries:
        life = calculate_battery_life(capacity, daily_consumption_mwh)
        ax.axvline(x=capacity, color='r', linestyle='--')
        ax.axhline(y=life, color='g', linestyle=':', alpha=0.5)
        
        # Add annotation
        ax.annotate(
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
    ax.axhspan(0, 30, alpha=0.1, color='red', label='< 30 days')
    ax.axhspan(30, 90, alpha=0.1, color='yellow', label='30-90 days')
    ax.axhspan(90, max(battery_lives), alpha=0.1, color='green', label='> 90 days')
    
    ax.set_xlabel('Battery Capacity (Wh)')
    ax.set_ylabel('Battery Life (Days)')
    ax.set_title('Battery Life vs. Capacity')
    ax.grid(True)
    ax.legend()
    
    plt.tight_layout()
    return fig

def plot_power_profile(consumption_data):
    """Create detailed power profile visualization with component breakdown."""
    fig, ax = plt.subplots(figsize=(15, 7))
    
    # Base timeline at 1-second resolution
    hours = 24
    seconds = hours * 3600
    timeline = np.zeros(seconds)
    
    # Add each component with different colors
    components = []
    
    # 1. Add continuous sensors (like IMU)
    if consumption_data["sensors"]["continuous"] > 0:
        continuous_power = consumption_data["sensors"]["continuous"] / hours
        base_timeline = np.ones(seconds) * continuous_power
        components.append(("Continuous Sensors", base_timeline, 'lightblue'))
    
    # 2. Add polled sensor events
    polled_timeline = np.zeros(seconds)
    if "sensors" in consumption_data and "details" in consumption_data["sensors"]:
        for sensor_id, details in consumption_data["sensors"]["details"].items():
            if details["operation_mode"] == "polled":
                interval = 3600 / consumption_data.get("base_frequency_per_hour", 60)
                for t in range(0, seconds, int(interval)):
                    if t + 100 < seconds:  # 100ms active time
                        polled_timeline[t:t+100] += details["average_mw"]
    if np.any(polled_timeline > 0):
        components.append(("Polled Sensors", polled_timeline, 'green'))
    
    # 3. Add GPS events
    gps_timeline = np.zeros(seconds)
    if "gps" in consumption_data["communications"]["details"]:
        gps = consumption_data["communications"]["details"]["gps"]
        interval = int(3600 / gps.get("updates", 24) * hours)
        for t in range(0, seconds, interval):
            if t + 30 < seconds:  # 30s active time
                gps_timeline[t:t+30] = COMMS_CONFIGS["GPS"]["power_modes"]["tracking"]["power"]
    if np.any(gps_timeline > 0):
        components.append(("GPS", gps_timeline, 'yellow'))
    
    # 4. Add Cellular events
    cellular_timeline = np.zeros(seconds)
    if "cellular" in consumption_data["communications"]["details"]:
        cellular = consumption_data["communications"]["details"]["cellular"]
        interval = int(24 * 3600 / cellular.get("sessions", 1))
        duration = int(cellular.get("duration_minutes", 1) * 60)
        for t in range(0, seconds, interval):
            if t + duration < seconds:
                cellular_timeline[t:t+duration] = COMMS_CONFIGS["CELLULAR"]["power_modes"]["active"]["power"]
    if np.any(cellular_timeline > 0):
        components.append(("Cellular", cellular_timeline, 'red'))
    
    # 5. Add LoRa events
    lora_timeline = np.zeros(seconds)
    if "lora" in consumption_data["communications"]["details"]:
        lora = consumption_data["communications"]["details"]["lora"]
        if lora.get("rx_enabled", False):
            lora_timeline += COMMS_CONFIGS["LORA"]["power_modes"]["rx"]["power"] * lora.get("rx_duty_cycle", 0.1)
        messages_per_day = 24 if lora.get("frequency_type") == "per_hour" else 1
        interval = int(24 * 3600 / (messages_per_day * lora.get("frequency", 1)))
        for t in range(0, seconds, interval):
            if t + 5 < seconds:  # 5s transmission time
                lora_timeline[t:t+5] = COMMS_CONFIGS["LORA"]["power_modes"]["tx"]["power"]
    if np.any(lora_timeline > 0):
        components.append(("LoRa", lora_timeline, 'purple'))
    
    # 6. Add Coprocessor events
    coprocessor_timeline = np.zeros(seconds)
    if consumption_data["coprocessor"].get("total", 0) > 0:
        coprocessor = consumption_data["coprocessor"]["details"]
        interval = int(24 * 3600 / coprocessor.get("windows", 1))
        duration = int(coprocessor.get("duration_minutes", 5) * 60)
        for t in range(0, seconds, interval):
            if t + duration < seconds:
                coprocessor_timeline[t:t+duration] = coprocessor.get("average_mw", 0)
    if np.any(coprocessor_timeline > 0):
        components.append(("Co-Processor", coprocessor_timeline, 'orange'))
    
    # Plot each component
    bottom = np.zeros(seconds)
    for name, timeline, color in components:
        # Downsample for plotting (1 point per minute)
        minutes = np.arange(24*60)
        minute_data = np.array([np.max(timeline[i*60:(i+1)*60]) for i in range(24*60)])
        ax.fill_between(minutes/60, bottom[::60], bottom[::60] + minute_data, 
                       label=name, alpha=0.5, color=color)
        bottom[::60] += minute_data
    
    # Add average line
    avg_power = consumption_data["average_power_mw"]
    ax.axhline(y=avg_power, color='black', linestyle='--', 
               label=f'Average ({avg_power:.1f} mW)')
    
    # Formatting
    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('Power Draw (mW)')
    ax.set_title('24-Hour Power Profile by Component')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.set_xlim(0, 24)
    
    # Set reasonable y-axis limits
    max_power = np.max(bottom)
    ax.set_ylim(0, max_power * 1.1)
    
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

def create_sensor_section(system_mode, default_config=None):
    """Create the sensor configuration section."""
    st.header("Sensor Configuration")
    st.markdown("---")
    
    sensors_config = {}
    
    # Create columns for better organization
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Sensor Selection")
        
        # IMU configuration
        st.write("**IMU Configuration (Always On)**")
        imu_enabled = default_config.get("LSM6DSV", {}).get("enabled", True) if default_config else True
        imu_mode = "typical" if imu_enabled else "low_power"
        
        sensors_config["LSM6DSV"] = {
            "enabled": imu_enabled,
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
                enabled = st.checkbox(
                    "Enable Sensor", 
                    value=default_config.get(sensor_id, {}).get("enabled", True) if default_config else True,
                    key=f"enable_{sensor_id}"
                )
                if enabled:
                    default_mode = "typical"
                    if default_config and sensor_id in default_config:
                        # Determine mode based on power values
                        default_mode = "low_power" if default_config[sensor_id]["active_power"] == sensor["low_power"]["active"] else "typical"
                    
                    power_mode = st.radio(
                        "Power Mode",
                        ["typical", "low_power"],
                        index=0 if default_mode == "typical" else 1,
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
    sensors_config['base_frequency_per_hour'] = st.number_input(
        "Base Sampling Frequency (times per hour):",
        value=default_config.get('base_frequency_per_hour', 60.0) if default_config else 60.0,
        min_value=1.0,
        step=1.0
    )
    
    sensors_config['base_duration_seconds'] = st.number_input(
        "Sampling Duration (seconds):",
        value=default_config.get('base_duration_seconds', 0.1) if default_config else 0.1,
        min_value=0.1,
        step=0.1,
        format="%.1f"
    )
    
    return sensors_config

def create_communications_section(default_config=None):
    """Create the communications configuration section."""
    st.header("Communications Configuration")
    st.markdown("---")
    
    comms_config = {"gps": {}, "cellular": {}, "lora": {}}
    
    # GPS Configuration
    with st.expander("GPS Configuration (MAX-M10S)"):
        default_gps = default_config.get('gps', {}) if default_config else {}
        gps_enabled = st.checkbox("Enable GPS", 
                                value=default_gps.get('enabled', True))
        
        if gps_enabled:
            col1, col2 = st.columns(2)
            with col1:
                freq = st.number_input(
                    "Update Frequency (times per hour):",
                    value=default_gps.get('frequency_per_hour',
                                        COMMS_CONFIGS["GPS"]["default_schedule"]["frequency"]),
                    min_value=0.1
                )
            with col2:
                duration = st.number_input(
                    "Fix Duration (seconds):",
                    value=default_gps.get('duration_seconds',
                                        COMMS_CONFIGS["GPS"]["default_schedule"]["duration"]),
                    min_value=1.0
                )
            
            comms_config["gps"] = {
                "enabled": True,
                "active_power": COMMS_CONFIGS["GPS"]["power_modes"]["tracking"]["power"],
                "frequency_per_hour": freq,
                "duration_seconds": duration
            }
        else:
            comms_config["gps"] = {"enabled": False}
    
    # Cellular Configuration
    with st.expander("Cellular Configuration (EG25-G)"):
        default_cellular = default_config.get('cellular', {}) if default_config else {}
        cellular_enabled = st.checkbox("Enable Cellular",
                                     value=default_cellular.get('enabled', True))
        
        if cellular_enabled:
            col1, col2 = st.columns(2)
            with col1:
                freq = st.number_input(
                    "Updates per Day:",
                    value=default_cellular.get('frequency_per_day',
                                             COMMS_CONFIGS["CELLULAR"]["default_schedule"]["frequency"]),
                    min_value=1
                )
            with col2:
                duration = st.number_input(
                    "Window Duration (minutes):",
                    value=default_cellular.get('duration_minutes',
                                             COMMS_CONFIGS["CELLULAR"]["default_schedule"]["duration"]),
                    min_value=1
                )
            
            comms_config["cellular"] = {
                "enabled": True,
                "active_power": COMMS_CONFIGS["CELLULAR"]["power_modes"]["active"]["power"],
                "frequency_per_day": freq,
                "duration_minutes": duration
            }
        else:
            comms_config["cellular"] = {"enabled": False}
    
    # LoRa Configuration
    with st.expander("LoRa Configuration (LLCC68)"):
        default_lora = default_config.get('lora', {}) if default_config else {}
        lora_enabled = st.checkbox("Enable LoRa",
                                 value=default_lora.get('enabled', False))
        
        if lora_enabled:
            col1, col2 = st.columns(2)
            with col1:
                freq_type = st.selectbox(
                    "Message Frequency Type:",
                    ["per_hour", "per_day"],
                    index=0 if default_lora.get('frequency_type', 'per_hour') == 'per_hour' else 1
                )
                freq = st.number_input(
                    f"Messages {freq_type}:",
                    value=default_lora.get('frequency', 1.0),
                    min_value=0.1
                )
            with col2:
                duration = st.number_input(
                    "Message Duration (seconds):",
                    value=default_lora.get('duration_seconds',
                                         COMMS_CONFIGS["LORA"]["power_modes"]["tx"]["typical_duration"]),
                    min_value=0.1
                )
                listen_enabled = st.checkbox("Enable Receive Mode",
                                          value=default_lora.get('listen_enabled', False))
            
            if listen_enabled:
                rx_duty_cycle = st.slider(
                    "Receive Duty Cycle (%):",
                    min_value=1,
                    max_value=100,
                    value=int(default_lora.get('rx_duty_cycle', 0.1) * 100)
                ) / 100.0
            else:
                rx_duty_cycle = 0
            
            comms_config["lora"] = {
                "enabled": True,
                "active_power": COMMS_CONFIGS["LORA"]["power_modes"]["tx"]["power"],
                "frequency_type": freq_type,
                "frequency": freq,
                "duration_seconds": duration,
                "listen_enabled": listen_enabled,
                "rx_duty_cycle": rx_duty_cycle
            }
        else:
            comms_config["lora"] = {"enabled": False}
    
    return comms_config
def create_coprocessor_section(default_config=None):
    """Create the coprocessor configuration section."""
    st.header("Co-Processor Configuration")
    st.markdown("---")
    
    coprocessor_enabled = st.checkbox("Enable Co-Processor", 
                                    value=default_config.get('enabled', False) if default_config else False)
    
    if coprocessor_enabled:
        col1, col2 = st.columns(2)
        
        with col1:
            coprocessor_type = st.selectbox(
                "Select Co-Processor:",
                list(COPROCESSOR_CONFIGS.keys()),
                index=0 if default_config is None else 
                    list(COPROCESSOR_CONFIGS.keys()).index(default_config.get('type', 'Jetson Orin Nano'))
            )
            
            power_mode = st.radio(
                "Power Mode:",
                ["typical", "max"],
                index=0 if default_config is None else 
                    1 if default_config.get('active_power') == COPROCESSOR_CONFIGS[coprocessor_type]["power_modes"]["max"]["active"] else 0
            )
        
        with col2:
            freq = st.number_input(
                "Processing Windows per Day:",
                value=default_config.get('frequency_per_day', 1) if default_config else 1,
                min_value=1,
                help="Number of processing sessions per day"
            )
            duration = st.number_input(
                "Window Duration (minutes):",
                value=default_config.get('duration_minutes', 5) if default_config else 5,
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
            "Battery Aging Factor (%)", 
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
        
        # Create and display power profile
        fig_power = plot_power_profile(consumption_data)
        st.pyplot(fig_power)
        
        # Add statistics
        col1, col2 = st.columns([3, 1])
        with col2:
            timeline = calculate_power_timeline(consumption_data)
            peak_power = np.max(timeline)
            min_power = np.min(timeline)
            avg_power = consumption_data["average_power_mw"]
            
            st.write("Power Statistics:")
            st.write(f"Peak Power: {peak_power:.2f} mW")
            st.write(f"Minimum Power: {min_power:.2f} mW")
            st.write(f"Average Power: {avg_power:.2f} mW")
            
            # Add duty cycle info
            active_time = np.sum(timeline > avg_power * 1.1)  # Time spent above average
            duty_cycle = (active_time / len(timeline)) * 100
            st.write(f"Active Duty Cycle: {duty_cycle:.1f}%")
def main():
    st.set_page_config(page_title="Battery Life Calculator", layout="wide")
    
    st.title("Battery Life Calculator")
    
    # Profile selection at the top
    st.header("Profile Selection")
    selected_profile = st.selectbox(
        "Select Configuration Profile:",
        ["Custom"] + list(DEFAULT_PROFILES.keys()),
        index=0
    )
    
    if selected_profile != "Custom":
        st.info(DEFAULT_PROFILES[selected_profile]["description"])
        if st.button("Load Profile"):
            st.session_state.current_profile = DEFAULT_PROFILES[selected_profile]
            st.rerun()
    
    # Get current profile if any
    current_profile = st.session_state.get('current_profile', {})
    
    # Create configuration sections
    st.markdown("---")
    sensors_config = create_sensor_section(
        system_mode="CONTINUOUS",  # Default mode
        default_config=current_profile.get("sensor_config") if current_profile else None
    )
    
    comms_config = create_communications_section(
        default_config=current_profile.get("comms_config") if current_profile else None
    )
    
    coprocessor_config = create_coprocessor_section(
        default_config=current_profile.get("coprocessor_config") if current_profile else None
    )
    
    battery_config = create_battery_section()
    
    if st.button("Calculate Battery Life"):
        consumption_data = calculate_total_consumption(
            sensors_config,
            comms_config,
            coprocessor_config,
            battery_config["derating"]
        )
        
        st.session_state.last_calculation = consumption_data
        display_results(consumption_data, battery_config)

if __name__ == "__main__":
    main()