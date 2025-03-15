"""
Hardware integration utilities for the Food Freshness Detector project.

This module provides functions for integrating gas sensors (e.g., MQ-series)
with the food freshness detection system using Raspberry Pi.
"""

import time
import argparse
from typing import Dict, List, Optional, Tuple, Union

try:
    import RPi.GPIO as GPIO
    import smbus
    RPI_AVAILABLE = True
except ImportError:
    RPI_AVAILABLE = False
    print("Warning: RPi.GPIO and/or smbus modules not available.")
    print("Hardware integration will be simulated.")


class MQGasSensor:
    """Class for interfacing with MQ-series gas sensors."""
    
    def __init__(self, 
                 pin: int = 0, 
                 adc_channel: int = 0,
                 sensor_type: str = 'MQ-3',
                 use_i2c: bool = True,
                 i2c_address: int = 0x48,
                 simulate: bool = not RPI_AVAILABLE):
        """
        Initialize the MQ gas sensor interface.
        
        Args:
            pin: GPIO pin number (for direct GPIO reading)
            adc_channel: ADC channel number (for I2C ADC)
            sensor_type: Type of MQ sensor (e.g., 'MQ-3', 'MQ-135')
            use_i2c: Whether to use I2C ADC for analog reading
            i2c_address: I2C address of the ADC
            simulate: Whether to simulate sensor readings
        """
        self.pin = pin
        self.adc_channel = adc_channel
        self.sensor_type = sensor_type
        self.use_i2c = use_i2c
        self.i2c_address = i2c_address
        self.simulate = simulate
        
        # Initialize hardware if not simulating
        if not self.simulate:
            if self.use_i2c:
                # Initialize I2C bus
                self.bus = smbus.SMBus(1)  # Use bus 1 for newer Raspberry Pi models
            else:
                # Initialize GPIO
                GPIO.setmode(GPIO.BCM)
                GPIO.setup(self.pin, GPIO.IN)
    
    def read_raw(self) -> int:
        """
        Read raw sensor value.
        
        Returns:
            Raw sensor reading
        """
        if self.simulate:
            # Simulate sensor reading
            import random
            return random.randint(100, 900)
        
        if self.use_i2c:
            # Read from I2C ADC
            try:
                # This is a simplified example for a generic ADC
                # For specific ADCs like ADS1115, you would use their respective libraries
                self.bus.write_byte(self.i2c_address, self.adc_channel)
                return self.bus.read_byte(self.i2c_address)
            except Exception as e:
                print(f"Error reading from I2C ADC: {e}")
                return 0
        else:
            # Read directly from GPIO (digital reading only)
            return GPIO.input(self.pin)
    
    def read_scaled(self, min_val: float = 0.0, max_val: float = 1.0) -> float:
        """
        Read scaled sensor value.
        
        Args:
            min_val: Minimum scaled value
            max_val: Maximum scaled value
            
        Returns:
            Scaled sensor reading
        """
        raw_value = self.read_raw()
        
        # Scale raw value (assuming raw range of 0-1023 for 10-bit ADC)
        scaled_value = min_val + (raw_value / 1023.0) * (max_val - min_val)
        
        return scaled_value
    
    def get_gas_concentration(self) -> Dict[str, float]:
        """
        Get gas concentration values.
        
        Returns:
            Dictionary of gas concentrations
        """
        # This is a simplified implementation
        # In a real application, you would use calibration curves and formulas
        # specific to each MQ sensor type to convert raw readings to ppm
        
        raw_value = self.read_raw()
        
        if self.sensor_type == 'MQ-3':
            # MQ-3 is sensitive to alcohol
            alcohol_ppm = raw_value * 0.15  # Simplified conversion
            return {'alcohol': alcohol_ppm}
        
        elif self.sensor_type == 'MQ-135':
            # MQ-135 is sensitive to air quality gases (CO2, NH3, NOx, etc.)
            co2_ppm = raw_value * 0.2  # Simplified conversion
            nh3_ppm = raw_value * 0.1  # Simplified conversion
            return {'co2': co2_ppm, 'nh3': nh3_ppm}
        
        else:
            # Generic reading
            return {'gas': raw_value * 0.1}
    
    def cleanup(self) -> None:
        """Clean up resources."""
        if not self.simulate and not self.use_i2c:
            GPIO.cleanup()


class SensorArray:
    """Class for managing multiple gas sensors."""
    
    def __init__(self, sensors: Optional[List[MQGasSensor]] = None):
        """
        Initialize the sensor array.
        
        Args:
            sensors: List of MQGasSensor instances
        """
        self.sensors = sensors or []
    
    def add_sensor(self, sensor: MQGasSensor) -> None:
        """
        Add a sensor to the array.
        
        Args:
            sensor: MQGasSensor instance
        """
        self.sensors.append(sensor)
    
    def read_all(self) -> List[Dict[str, float]]:
        """
        Read all sensors.
        
        Returns:
            List of sensor readings
        """
        return [sensor.get_gas_concentration() for sensor in self.sensors]
    
    def get_feature_vector(self) -> List[float]:
        """
        Get a feature vector for model input.
        
        Returns:
            List of sensor values as a feature vector
        """
        # Flatten all sensor readings into a single feature vector
        feature_vector = []
        
        for reading in self.read_all():
            feature_vector.extend(reading.values())
        
        return feature_vector
    
    def cleanup(self) -> None:
        """Clean up all sensors."""
        for sensor in self.sensors:
            sensor.cleanup()


def create_default_sensor_array(simulate: bool = not RPI_AVAILABLE) -> SensorArray:
    """
    Create a default sensor array configuration.
    
    Args:
        simulate: Whether to simulate sensor readings
        
    Returns:
        Configured SensorArray instance
    """
    # Create sensor array
    sensor_array = SensorArray()
    
    # Add MQ-3 sensor (alcohol detection)
    mq3_sensor = MQGasSensor(
        pin=17,
        adc_channel=0,
        sensor_type='MQ-3',
        use_i2c=True,
        simulate=simulate
    )
    sensor_array.add_sensor(mq3_sensor)
    
    # Add MQ-135 sensor (air quality)
    mq135_sensor = MQGasSensor(
        pin=18,
        adc_channel=1,
        sensor_type='MQ-135',
        use_i2c=True,
        simulate=simulate
    )
    sensor_array.add_sensor(mq135_sensor)
    
    return sensor_array


def main():
    """Test the gas sensor functionality."""
    parser = argparse.ArgumentParser(description='Test gas sensors for Food Freshness Detector')
    parser.add_argument('--simulate', action='store_true',
                        help='Simulate sensor readings')
    parser.add_argument('--duration', type=int, default=10,
                        help='Duration to read sensors (in seconds)')
    parser.add_argument('--interval', type=float, default=1.0,
                        help='Interval between readings (in seconds)')
    
    args = parser.parse_args()
    
    # Create sensor array
    sensor_array = create_default_sensor_array(simulate=args.simulate)
    
    try:
        print(f"Reading sensors for {args.duration} seconds...")
        end_time = time.time() + args.duration
        
        while time.time() < end_time:
            # Read all sensors
            readings = sensor_array.read_all()
            
            # Print readings
            print("\nSensor Readings:")
            for i, reading in enumerate(readings):
                print(f"Sensor {i+1} ({sensor_array.sensors[i].sensor_type}):")
                for gas, value in reading.items():
                    print(f"  {gas}: {value:.2f} ppm")
            
            # Get feature vector
            feature_vector = sensor_array.get_feature_vector()
            print(f"\nFeature Vector: {feature_vector}")
            
            # Wait for next reading
            time.sleep(args.interval)
        
    except KeyboardInterrupt:
        print("\nStopped by user")
    
    finally:
        # Clean up
        sensor_array.cleanup()
        print("\nSensor cleanup completed")


if __name__ == "__main__":
    main()
