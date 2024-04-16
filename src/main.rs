use rand::{rngs::ThreadRng, thread_rng, Rng};
use std::{mem::size_of, path::PathBuf, str::FromStr, usize};

const VIDEO_WIDTH: usize = 64;
const VIDEO_HEIGHT: usize = 32;
const FONTSET_START_ADDRESS: usize = 0x50;

const START_ADDRESS: usize = 0x200;
const FONTSET_SIZE: usize = 80;

const FONTSET: [u8; FONTSET_SIZE] = [
    0xF0, 0x90, 0x90, 0x90, 0xF0, // 0
    0x20, 0x60, 0x20, 0x20, 0x70, // 1
    0xF0, 0x10, 0xF0, 0x80, 0xF0, // 2
    0xF0, 0x10, 0xF0, 0x10, 0xF0, // 3
    0x90, 0x90, 0xF0, 0x10, 0x10, // 4
    0xF0, 0x80, 0xF0, 0x10, 0xF0, // 5
    0xF0, 0x80, 0xF0, 0x90, 0xF0, // 6
    0xF0, 0x10, 0x20, 0x40, 0x40, // 7
    0xF0, 0x90, 0xF0, 0x90, 0xF0, // 8
    0xF0, 0x90, 0xF0, 0x10, 0xF0, // 9
    0xF0, 0x90, 0xF0, 0x90, 0x90, // A
    0xE0, 0x90, 0xE0, 0x90, 0xE0, // B
    0xF0, 0x80, 0x80, 0x80, 0xF0, // C
    0xE0, 0x90, 0x90, 0x90, 0xE0, // D
    0xF0, 0x80, 0xF0, 0x80, 0xF0, // E
    0xF0, 0x80, 0xF0, 0x80, 0x80, // F
];

#[derive(Debug)]
struct Chip8 {
    register: [u8; 16],
    memory: [u8; 4096],
    index: u16,
    pc: u16,
    stack: [u16; 16],
    sp: u16,
    delay_timer: u8,
    sound_timer: u8,
    keypad: [bool; 16],
    video: [u8; 64 * 32],
    opcode: u16,
    rng: ThreadRng,
}

impl Default for Chip8 {
    fn default() -> Self {
        let mut memory = [0u8; 4096];

        // set font set
        for (i, el) in FONTSET.into_iter().enumerate() {
            memory[FONTSET_START_ADDRESS + i] = el;
        }

        let r = thread_rng();

        Self {
            register: Default::default(),
            rng: r,
            memory,
            index: Default::default(),
            // Pointer Counter
            pc: START_ADDRESS as u16,
            stack: Default::default(),
            // Stack Pointer
            sp: Default::default(),
            delay_timer: Default::default(),
            sound_timer: Default::default(),
            keypad: Default::default(),
            video: [0u8; 64 * 32],
            opcode: Default::default(),
        }
    }
}

impl Chip8 {
    fn load_rom(filepath: PathBuf) -> std::io::Result<Self> {
        let mut chip = Chip8::default();

        let buffer = std::fs::read(filepath)?;

        // Load the ROM contents into the Chip8's memory, starting at 0x200
        for (idx, byte) in buffer.into_iter().enumerate() {
            chip.memory[START_ADDRESS + idx] = byte;
        }

        Ok(chip)
    }

    fn set_key(&mut self, key: usize, state: bool) {
        self.keypad[key] = state;
    }

    fn cycle(&mut self) {
        self.opcode = ((self.memory[self.pc as usize] as u16) << 8)
            | self.memory[(self.pc as usize) + 1] as u16;
        self.pc += 2;

        self.table(self.opcode & 0xF000 >> 12);

        if self.delay_timer > 0 {
            self.delay_timer -= 1;
        }

        if self.sound_timer > 0 {
            self.sound_timer -= 1;
        }
    }

    fn table(&mut self, code: u16) {
        match code {
            0x0 => match code & 0x000F {
                0x0 => self.op_00_e0(),
                0xE => self.op_00_ee(),
                _ => {}
            },
            0x1 => self.op_1nnn(),
            0x2 => self.op_2nnn(),
            0x3 => self.op_3xkk(),
            0x4 => self.op_4xkk(),
            0x5 => self.op_5xy0(),
            0x6 => self.op_6xkk(),
            0x7 => self.op_7xkk(),
            0x8 => match code & 0x000F {
                0x0 => self.op_8xy0(),
                0x1 => self.op_8xy1(),
                0x2 => self.op_8xy2(),
                0x3 => self.op_8xy3(),
                0x4 => self.op_8xy4(),
                0x5 => self.op_8xy5(),
                0x6 => self.op_8xy6(),
                0x7 => self.op_8xy7(),
                0xE => self.op_8xye(),
                _ => {}
            },
            0x9 => self.op_9xy0(),
            0xA => self.op_annn(),
            0xB => self.op_bnnn(),
            0xC => self.op_cxkk(),
            0xD => self.op_dxyn(),
            0xE => match code & 0x000F {
                0x1 => self.op_exa1(),
                0xE => self.op_ex9e(),
                _ => {}
            },
            0xF => match code & 0x00FF {
                0x07 => self.op_fx07(),
                0x0A => self.op_fx0a(),
                0x15 => self.op_fx15(),
                0x18 => self.op_fx18(),
                0x1E => self.op_fx1e(),
                0x29 => self.op_fx29(),
                0x33 => self.op_fx33(),
                0x55 => self.op_fx55(),
                0x65 => self.op_fx65(),
                _ => {}
            },
            _ => {}
        }
    }

    fn rand_byte(&mut self) -> u8 {
        self.rng.gen_range(0..255)
    }

    /// Clear the display.
    fn op_00_e0(&mut self) {
        self.video = [0u8; 64 * 32];
    }
    /// Return from a subroutine.
    ///
    /// The top of the stack has the address of one instruction past the one that called the subroutine,
    /// so we can put that back into the PC. Note that this overwrites our preemptive pc += 2 earlier.
    fn op_00_ee(&mut self) {
        self.sp -= 1;

        self.pc = self.stack[self.sp as usize];
    }
    /// Jump to location nnn.
    ///
    /// The interpreter sets the program counter to nnn.
    fn op_1nnn(&mut self) {
        self.pc = self.opcode & 0x0FFF;
    }
    /// Call subroutine at nnn.
    ///
    /// When we call a subroutine, we want to return eventually, so we put the current PC onto the top of the stack.
    /// Remember that we did pc += 2 in Cycle(), so the current PC holds the next instruction after this CALL,
    /// which is correct. We don’t want to return to the CALL instruction because it would be an infinite loop of CALLs and RETs.
    fn op_2nnn(&mut self) {
        let address = self.opcode & 0x0FFF;

        self.stack[self.sp as usize] = self.pc;
        self.sp += 1;
        self.pc = address;
    }
    /// Skip next instruction if Vx = kk.
    fn op_3xkk(&mut self) {
        let vx = ((self.opcode & 0x0F00) >> 8) as usize;
        let byte = (self.opcode & 0x00FF) as u8;

        if self.register[vx] == byte {
            self.pc += 2;
        }
    }
    /// Skip next instruction if Vx != kk.
    fn op_4xkk(&mut self) {
        let vx = ((self.opcode & 0x0F00) >> 8) as usize;
        let byte = (self.opcode & 0x00FF) as u8;

        if self.register[vx] != byte {
            self.pc += 2;
        }
    }
    /// Skip next instruction if Vx = Vy.
    fn op_5xy0(&mut self) {
        let vx = ((self.opcode & 0x0F00) >> 8) as usize;
        let vy = ((self.opcode & 0x00F0) >> 4) as usize;

        if self.register[vx] == self.register[vy] {
            self.pc += 2;
        }
    }
    /// Set Vx = kk.
    fn op_6xkk(&mut self) {
        let vx = ((self.opcode & 0x0F00) > 8) as usize;
        let byte = (self.opcode & 0x00FF) as u8;

        self.register[vx] = byte;
    }
    /// Set Vx = Vx + kk.
    fn op_7xkk(&mut self) {
        let vx = ((self.opcode & 0x0F00) >> 8) as usize;
        let byte = (self.opcode & 0x00FF) as u8;

        self.register[vx] += byte;
    }
    /// Set Vx = Vy.
    fn op_8xy0(&mut self) {
        let vx = ((self.opcode & 0x0F00) >> 8) as usize;
        let vy = ((self.opcode & 0x00F0) >> 4) as usize;

        self.register[vx] = self.register[vy];
    }
    /// Set Vx = Vx OR Vy.
    fn op_8xy1(&mut self) {
        let vx = ((self.opcode & 0x0F00) >> 8) as usize;
        let vy = ((self.opcode & 0x00F0) >> 4) as usize;

        self.register[vx] |= self.register[vy];
    }
    /// Set Vx = Vx AND Vy.
    fn op_8xy2(&mut self) {
        let vx = ((self.opcode & 0x0F00) >> 8) as usize;
        let vy = ((self.opcode & 0x00F0) >> 4) as usize;

        self.register[vx] &= self.register[vy];
    }
    /// Set Vx = Vx XOR Vy.
    fn op_8xy3(&mut self) {
        let vx = ((self.opcode & 0x0F00) >> 8) as usize;
        let vy = ((self.opcode & 0x00F0) >> 4) as usize;

        self.register[vx] ^= self.register[vy];
    }
    /// Set Vx = Vx + Vy, set VF = carry.
    ///
    /// The values of Vx and Vy are added together.
    /// If the result is greater than 8 bits (i.e., > 255,) VF is set to 1, otherwise 0.
    /// Only the lowest 8 bits of the result are kept, and stored in Vx.
    fn op_8xy4(&mut self) {
        let vx = ((self.opcode & 0x0F00) >> 8) as usize;
        let vy = ((self.opcode & 0x00F0) >> 4) as usize;

        // may be larger then a u8 so we cast to u16 to not overflow
        let sum = (self.register[vx]) as u16 + (self.register[vy]) as u16;

        self.register[0xF] = if sum > 255 { 1 } else { 0 };

        self.register[vx] = (sum & 0xFF) as u8;
    }
    /// Set Vx = Vx - Vy, set VF = NOT borrow.
    ///
    /// If Vx > Vy, then VF is set to 1, otherwise 0.
    /// Then Vy is subtracted from Vx, and the results stored in Vx.
    fn op_8xy5(&mut self) {
        let vx = ((self.opcode & 0x0F00) >> 8) as usize;
        let vy = ((self.opcode & 0x00F0) >> 4) as usize;

        self.register[0xF] = if self.register[vx] > self.register[vy] {
            1
        } else {
            0
        };

        self.register[vx] -= self.register[vy];
    }
    /// Set Vx = Vx SHR 1.
    ///
    /// If the least-significant bit of Vx is 1, then VF is set to 1, otherwise 0.
    /// Then Vx is divided by 2.
    fn op_8xy6(&mut self) {
        let vx = ((self.opcode & 0x0F00) >> 8) as usize;

        self.register[0xF] = self.register[vx] & 0x1;

        self.register[vx] >>= 1;
    }

    /// Set Vx = Vy - Vx, set VF = NOT borrow.
    ///
    /// Vy > Vx, then VF is set to 1, otherwise 0. Then Vx is subtracted from Vy,
    /// and the results stored in Vx.
    fn op_8xy7(&mut self) {
        let vx = ((self.opcode & 0x0F00) >> 8) as usize;
        let vy = ((self.opcode & 0x00F0) >> 4) as usize;

        self.register[0xF] = if self.register[vy] > self.register[vx] {
            1
        } else {
            0
        };

        self.register[vx] = self.register[vy] - self.register[vx];
    }

    /// Set Vx = Vx SHL 1.
    ///
    /// If the most-significant bit of Vx is 1, then VF is set to 1, otherwise to 0.
    /// Then Vx is multiplied by 2.
    fn op_8xye(&mut self) {
        let vx = ((self.opcode & 0x0F00) >> 8) as usize;

        self.register[0xF] = (self.register[vx] & 0x80) >> 7;

        self.register[vx] <<= 1;
    }
    /// Skip next instruction if Vx != Vy.
    fn op_9xy0(&mut self) {
        let vx = ((self.opcode & 0x0F00) >> 8) as usize;
        let vy = ((self.opcode & 0x00F0) >> 4) as usize;

        if self.register[vx] != self.register[vy] {
            self.pc += 2;
        }
    }
    /// Set I = nnn.
    fn op_annn(&mut self) {
        self.index = self.opcode & 0x0FFF;
    }
    /// Jump to location nnn + V0.
    fn op_bnnn(&mut self) {
        let address = self.opcode & 0x0FFF;
        self.pc = (self.register[0] as u16) + address;
    }
    /// Set Vx = random byte AND kk.
    fn op_cxkk(&mut self) {
        let vx = ((self.opcode & 0x0F00) >> 8) as usize;
        let byte = (self.opcode & 0x00FF) as u8;

        self.register[vx] = self.rand_byte() & byte;
    }
    /// Display n-byte sprite starting at memory location I at (Vx, Vy), set VF = collision.
    fn op_dxyn(&mut self) {
        let vx = ((self.opcode & 0x0F00) >> 8) as usize;
        let vy = ((self.opcode & 0x00F0) >> 4) as usize;
        let height = self.opcode & 0x000F;

        let xpos = (self.register[vx] as usize) % VIDEO_WIDTH;
        let ypos = (self.register[vy] as usize) % VIDEO_HEIGHT;

        self.register[0xF] = 0;

        for row in 0..height {
            let sprite_byte = self.memory[(self.index + row) as usize];
            for col in 0..8 {
                let sprite_pixel = sprite_byte & (0x80 >> col);
                let screen_pos = (ypos + (row as usize)) * VIDEO_WIDTH + (xpos + col);
                let screen_pixel = self.video[screen_pos];

                if sprite_pixel != 0 {
                    if screen_pixel == 0xFF {
                        self.register[0xF] = 1;
                    }

                    self.video[screen_pos] ^= 0xFF;
                }
            }
        }
    }

    /// Skip next instruction if key with the value of Vx is pressed.
    fn op_ex9e(&mut self) {
        let vx = ((self.opcode & 0x0F00) >> 8) as usize;
        let key = self.register[vx] as usize;

        if let Some(true) = self.keypad.get(key) {
            self.pc += 2;
        }
    }
    /// Skip next instruction if key with the value of Vx is not pressed.
    fn op_exa1(&mut self) {
        let vx = ((self.opcode & 0x0F00) >> 8) as usize;
        let key = self.register[vx] as usize;

        if let Some(false) = self.keypad.get(key) {
            self.pc += 2;
        }
    }

    /// Set Vx = delay timer value.
    fn op_fx07(&mut self) {
        let vx = ((self.opcode & 0x0F00) >> 8) as usize;

        self.register[vx] = self.delay_timer;
    }

    /// Wait for a key press, store the value of the key in Vx.
    fn op_fx0a(&mut self) {
        let vx = ((self.opcode & 0x0F00) >> 8) as usize;

        if let Some(idx) = self.keypad.iter().position(|&x| x) {
            self.register[vx] = idx as u8;
        } else {
            self.pc -= 2;
        }
    }

    /// Set delay timer = Vx.
    fn op_fx15(&mut self) {
        let vx = ((self.opcode & 0x0F00) >> 8) as usize;

        self.delay_timer = self.register[vx];
    }
    /// Set sound timer = Vx.
    fn op_fx18(&mut self) {
        let vx = ((self.opcode & 0x0F00) >> 8) as usize;

        self.sound_timer = self.register[vx];
    }
    /// Set I = I + Vx.
    fn op_fx1e(&mut self) {
        let vx = ((self.opcode & 0x0F00) >> 8) as usize;

        self.index += self.register[vx] as u16;
    }
    /// Set I = location of sprite for digit Vx.
    fn op_fx29(&mut self) {
        let vx = ((self.opcode & 0x0F00) >> 8) as usize;
        let digit = self.register[vx] as u16;
        self.index = (FONTSET_START_ADDRESS as u16) + (5 * digit);
    }

    /// Store BCD representation of Vx in memory locations I, I+1, and I+2.
    ///
    /// The interpreter takes the decimal value of Vx, and places the hundreds digit in memory at location in I,
    /// the tens digit at location I+1, and the ones digit at location I+2.
    fn op_fx33(&mut self) {
        let vx = ((self.opcode & 0x0F00) >> 8) as usize;
        let mut value = self.register[vx];

        self.memory[(self.index + 2) as usize] = value % 10;
        value /= 10;

        self.memory[(self.index + 1) as usize] = value % 10;
        value /= 10;

        self.memory[self.index as usize] = value % 10;
    }

    /// Store registers V0 through Vx in memory starting at location I.
    fn op_fx55(&mut self) {
        let vx = ((self.opcode & 0x0F00) >> 8) as usize;

        for i in 0..=vx {
            self.memory[(self.index + 1) as usize] = self.register[i];
        }
    }

    /// Read registers V0 through Vx from memory starting at location I.
    fn op_fx65(&mut self) {
        let vx = ((self.opcode & 0x0F00) >> 8) as usize;

        for i in 0..=vx {
            self.register[i] = self.memory[(self.index + 1) as usize];
        }
    }
}

struct SDLRenderer {
    canvas: sdl2::render::Canvas<sdl2::video::Window>,
    event_pump: sdl2::EventPump,
}

impl SDLRenderer {
    pub fn new(width: u32, height: u32) -> Result<Self, String> {
        let ctx = sdl2::init()?;
        let vid_sys = ctx.video()?;

        let window = vid_sys
            .window("Chip 8", width, height)
            .position_centered()
            .build()
            .map_err(|x| x.to_string())?;

        let canvas = window
            .into_canvas()
            .accelerated()
            .build()
            .map_err(|e| e.to_string())?;

        let event_pump = ctx.event_pump()?;

        Ok(Self { canvas, event_pump })
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

    pub fn render(
        &mut self,
        pitch: usize,
        pixel_data: &[u8],
        texture: &mut sdl2::render::Texture,
    ) -> Result<(), String> {
        texture
            .update(None, pixel_data, pitch)
            .map_err(|e| e.to_string())?;
        // update texture
        self.canvas.clear();

        self.canvas
            .copy(texture, None, None)
            .map_err(|e| e.to_string())?;
        // render copy
        self.canvas.present();

        Ok(())
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let video_scale = args
        .get(1)
        .expect("Failed to get Scale")
        .parse::<u32>()
        .expect("Failed to parse video scale.");
    let cycle_delay = std::time::Duration::from_millis(
        args.get(2)
            .expect("Failed to get delay.")
            .parse::<u64>()
            .expect("Failed to parse delay."),
    );

    let file_path = PathBuf::from_str(args.get(3).expect("Failed to get ROM Path"))
        .expect("Failed to parse rom path");

    if !file_path.exists() || !file_path.is_file() {
        panic!("File does not exists")
    }

    let mut platform = SDLRenderer::new(
        (VIDEO_WIDTH as u32) * video_scale,
        (VIDEO_HEIGHT) as u32 * video_scale,
    )
    .expect("Failed to init sdl2");

    let binding = platform.canvas.texture_creator();
    let mut texture = binding
        .create_texture_streaming(
            Some(sdl2::pixels::PixelFormatEnum::RGBA8888),
            VIDEO_WIDTH as u32,
            VIDEO_HEIGHT as u32,
        )
        .expect("Failed to construct texture");

    let mut chip = Chip8::load_rom(file_path).expect("Failed to load rom");
    let video_pitch = size_of::<u32>() * VIDEO_WIDTH;

    let mut last_cycle_time = std::time::Instant::now();

    'running: loop {
        let current_time = std::time::Instant::now();
        let dt = current_time.duration_since(last_cycle_time);

        for event in platform.event_pump.poll_iter() {
            if let Some((key, state)) = SDLRenderer::process_input(event) {
                if key == 1024 {
                    break 'running;
                }

                chip.set_key(key, state);
            }
        }

        if dt.gt(&cycle_delay) {
            last_cycle_time = current_time;
            chip.cycle();

            platform
                .render(video_pitch, &chip.video, &mut texture)
                .expect("Failed to render.");
        }
    }
}
