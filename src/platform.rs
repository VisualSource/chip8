use sdl2::render::{Canvas, Texture, TextureCreator};
use sdl2::video::{Window, WindowContext};
use std::cell::RefCell;

// https://users.rust-lang.org/t/rust-sdl2-and-raw-textures-help/45636/24?page=2
pub struct SDLRenderer {
    ctx: sdl2::Sdl,
    canvas: Canvas<Window>,
    creator: TextureCreator<WindowContext>,
    texture: RefCell<Texture<'static>>,

    pub event_pump: sdl2::EventPump,
}

impl SDLRenderer {
    pub fn new(width: u32, height: u32) -> Result<Self, String> {
        let ctx = sdl2::init()?;
        let video_subsystem = ctx.video()?;

        let window = video_subsystem
            .window("Chip 8", width, height)
            .position_centered()
            .resizable()
            .build()
            .map_err(|x| x.to_string())?;

        let canvas = window
            .into_canvas()
            .accelerated()
            .build()
            .map_err(|e| e.to_string())?;

        let creator = canvas.texture_creator();

        let texture = creator
            .create_texture_streaming(Some(sdl2::pixels::PixelFormatEnum::RGBA8888), 64, 32)
            .map_err(|e| e.to_string())?;

        // create_texture_streaming returns a texture struct with a life time that says that it will only last
        // inside this function, which makes it hard to store a reference to the texture
        // in this struct. So we need this unsafe block and wrap it in a RefCell to get around this issue
        let texture = unsafe { std::mem::transmute::<_, Texture<'static>>(texture) };

        let event_pump = ctx.event_pump()?;

        Ok(Self {
            canvas,
            event_pump,
            ctx,
            creator,
            texture: RefCell::new(texture),
        })
    }

    pub fn render(&mut self, pitch: usize, pixel_data: &[u32]) -> Result<(), String> {
        // convert the u32 array into a u8 array so we can render correctly.
        let buffer = unsafe {
            std::slice::from_raw_parts(pixel_data.as_ptr() as *const u8, pixel_data.len() * 4)
        };

        let mut texture = self.texture.borrow_mut();

        texture
            .update(None, buffer, pitch)
            .map_err(|e| e.to_string())?;
        // update texture
        self.canvas.clear();

        self.canvas
            .copy(&texture, None, None)
            .map_err(|e| e.to_string())?;

        // render copy
        self.canvas.present();

        Ok(())
    }

    pub fn process_input(event: sdl2::event::Event) -> Option<(usize, bool)> {
        match event {
            sdl2::event::Event::Quit { .. } => Some((1024, false)),
            sdl2::event::Event::KeyDown {
                keycode: Some(code),
                ..
            } => {
                let id = match code {
                    sdl2::keyboard::Keycode::Num1 => 1,
                    sdl2::keyboard::Keycode::Num2 => 2,
                    sdl2::keyboard::Keycode::Num3 => 3,
                    sdl2::keyboard::Keycode::Num4 => 0xC,
                    sdl2::keyboard::Keycode::A => 7,
                    sdl2::keyboard::Keycode::C => 0xB,
                    sdl2::keyboard::Keycode::D => 9,
                    sdl2::keyboard::Keycode::E => 6,
                    sdl2::keyboard::Keycode::F => 0xE,
                    sdl2::keyboard::Keycode::Q => 4,
                    sdl2::keyboard::Keycode::R => 0xD,
                    sdl2::keyboard::Keycode::S => 8,
                    sdl2::keyboard::Keycode::V => 0xF,
                    sdl2::keyboard::Keycode::W => 5,
                    sdl2::keyboard::Keycode::X => 0,
                    sdl2::keyboard::Keycode::Z => 0xA,
                    _ => return None,
                };
                Some((id, true))
            }
            sdl2::event::Event::KeyUp {
                keycode: Some(code),
                ..
            } => {
                let id = match code {
                    sdl2::keyboard::Keycode::Num1 => 1,
                    sdl2::keyboard::Keycode::Num2 => 2,
                    sdl2::keyboard::Keycode::Num3 => 3,
                    sdl2::keyboard::Keycode::Num4 => 0xC,
                    sdl2::keyboard::Keycode::A => 7,
                    sdl2::keyboard::Keycode::C => 0xB,
                    sdl2::keyboard::Keycode::D => 9,
                    sdl2::keyboard::Keycode::E => 6,
                    sdl2::keyboard::Keycode::F => 0xE,
                    sdl2::keyboard::Keycode::Q => 4,
                    sdl2::keyboard::Keycode::R => 0xD,
                    sdl2::keyboard::Keycode::S => 8,
                    sdl2::keyboard::Keycode::V => 0xF,
                    sdl2::keyboard::Keycode::W => 5,
                    sdl2::keyboard::Keycode::X => 0,
                    sdl2::keyboard::Keycode::Z => 0xA,
                    _ => return None,
                };

                Some((id, false))
            }
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::mem::size_of;

    use super::SDLRenderer;

    #[test]
    fn test_render() {
        let mut render = SDLRenderer::new(64 * 10, 32 * 10).expect("Failed to build sdl");

        let pitch = size_of::<u32>() * 64;

        let mut buffer = vec![0u32; 64 * 32];

        buffer[0] = 0xFFFFFFFF;
        buffer[1] = 0xFFFFFFFF;
        buffer[2] = 0xFFFFFFFF;
        buffer[3] = 0xFFFFFFFF;

        buffer[64] = 0xFFFFFFFF;

        buffer[128] = 0xFFFFFFFF;
        buffer[129] = 0xFFFFFFFF;
        buffer[130] = 0xFFFFFFFF;
        buffer[131] = 0xFFFFFFFF;

        buffer[192] = 0xFFFFFFFF;

        buffer[256] = 0xFFFFFFFF;

        loop {
            render.render(pitch, &buffer).expect("Failed to render");
        }
    }
}
