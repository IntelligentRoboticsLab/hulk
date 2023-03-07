use std::time::{Duration, SystemTime, UNIX_EPOCH};

use module_derive::{module, require_some};
use types::{SensorData, SolePressure};

use std::fs::File;
use std::io::BufReader;
use rodio::{Decoder, OutputStream, source::Source};


use crate::control::filtering::greater_than_with_hysteresis;

pub struct GroundContactDetector {
    last_has_pressure: bool,
    last_time_switched: SystemTime,
    has_ground_contact: bool,
    sound_played: bool,
}

#[module(control)]
#[input(path = sensor_data, data_type = SensorData)]
#[input(path = sole_pressure, data_type = SolePressure)]
#[parameter(path = control.ground_contact_detector.pressure_threshold, data_type = f32)]
#[parameter(path = control.ground_contact_detector.hysteresis, data_type = f32)]
#[parameter(path = control.ground_contact_detector.timeout, data_type = Duration)]
#[main_output(data_type = bool, name = has_ground_contact)]
impl GroundContactDetector {}

impl GroundContactDetector {
    fn new(_context: NewContext) -> anyhow::Result<Self> {
        Ok(Self {
            last_has_pressure: false,
            last_time_switched: UNIX_EPOCH,
            has_ground_contact: false,
            sound_played: false,
        })
    }

    fn cycle(&mut self, context: CycleContext) -> anyhow::Result<MainOutputs> {
        let sensor_data = require_some!(context.sensor_data);
        let sole_pressure = require_some!(context.sole_pressure);
        let has_pressure = greater_than_with_hysteresis(
            self.last_has_pressure,
            sole_pressure.total(),
            *context.pressure_threshold,
            *context.hysteresis,
        );
        if self.last_has_pressure != has_pressure {
            self.last_time_switched = sensor_data.cycle_info.start_time;
        }
        if sensor_data
            .cycle_info
            .start_time
            .duration_since(self.last_time_switched)
            .expect("Time ran backwards")
            > *context.timeout
        {
            self.has_ground_contact = has_pressure;
        }
        self.last_has_pressure = has_pressure;
        
        if self.has_ground_contact{
            self.sound_played = false;
        }

        if !self.has_ground_contact && !self.sound_played{
            // Get a output stream handle to the default physical sound device
            let (_stream, stream_handle) = OutputStream::try_default().unwrap();
            // Load a sound from a file, using a path relative to Cargo.toml
            let file = BufReader::new(File::open("etc/sounds/weeeee.wav").unwrap());
            // Decode that sound file into a source
            let source = Decoder::new(file).unwrap();
            // Play the sound directly on the device
            stream_handle.play_raw(source.convert_samples()).ok(); 
            std::thread::sleep(std::time::Duration::from_secs(2));
                self.sound_played = true;  
            }

        Ok(MainOutputs {
            has_ground_contact: Some(self.has_ground_contact),
        })
    }
}
