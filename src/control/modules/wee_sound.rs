use std::time::{Duration};

use module_derive::{module, require_some};

use kira::{
	manager::{
		AudioManager, AudioManagerSettings,
		backend::cpal::CpalBackend,
	},
	sound::static_sound::{StaticSoundData, StaticSoundSettings},
        tween::Tween,
};


pub struct PlaySound {
    sound_played: bool,
}

#[module(control)]
#[input(path = has_ground_contact, data_type = bool)]
#[main_output(data_type = bool, name = has_ground_contact)]
impl PlaySound {}

impl PlaySound {
    fn new(_context: NewContext) -> anyhow::Result<Self> {
        Ok(Self {
            sound_played: true,
        })
    }

    fn cycle(&mut self, context: CycleContext) -> anyhow::Result<MainOutputs> {
        let has_ground_contact = require_some!(context.has_ground_contact);
        
        if *has_ground_contact{
            self.sound_played = false;
        }

        if !has_ground_contact && !self.sound_played{
            let mut manager = AudioManager::<CpalBackend>::new(AudioManagerSettings::default()).unwrap();
            let sound_data = StaticSoundData::from_file("etc/sounds/weeeee.wav", StaticSoundSettings::new()).unwrap();
            let mut sound =  manager.play(sound_data).unwrap();
            sound.set_volume(
                0.1,
                Tween {
                    duration: Duration::from_secs(0),
                    ..Default::default()
                },
            ).unwrap();
            self.sound_played = true;  
            }

        Ok(MainOutputs {
            has_ground_contact: Some(*has_ground_contact),
        })
    }
}
