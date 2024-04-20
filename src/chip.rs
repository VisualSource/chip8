use rand::{rngs::ThreadRng, thread_rng, Rng};
use std::{borrow::BorrowMut, path::PathBuf};

pub const VIDEO_WIDTH: usize = 64;
pub const VIDEO_HEIGHT: usize = 32;

const MEMORY_SIZE: usize = 4096; // 4KB memory
const REGISTER_SIZE: usize = 16;
const FLAGS_SIZE: usize = 16;
const KEYPAD_SIZE: usize = 16;
const STACK_SIZE: usize = 16;

const START_ADDRESS: u16 = 0x200;

const FONTSET_SIZE: usize = 80;
const FONTSET_START_ADDRESS: usize = 0x50;

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

pub struct Chip8 {
    register: [u8; REGISTER_SIZE],
    memory: [u8; MEMORY_SIZE],
    flags: [u8; FLAGS_SIZE],
    index: u16,
    /// Program Counter
    ///
    /// holds the address of the next instruction to execute.
    pc: u16,
    // the stack, for function and stuff
    stack: [u16; STACK_SIZE],
    // where we are on the statck
    sp: u16,

    // timer
    delay_timer: u8,
    // timer for sound
    sound_timer: u8,
    // the keys we are watching
    keypad: [bool; KEYPAD_SIZE],

    // gfx buffer
    video: [u32; VIDEO_WIDTH * VIDEO_HEIGHT],

    // just used for printing opcode for
    opcode: String,

    rng: ThreadRng,
    render: bool,
}

impl Default for Chip8 {
    fn default() -> Self {
        let mut memory = [0u8; MEMORY_SIZE];

        // set font set
        for (i, byte) in FONTSET.into_iter().enumerate() {
            memory[FONTSET_START_ADDRESS + i] = byte;
        }

        Self {
            rng: thread_rng(),
            memory,
            pc: START_ADDRESS,
            video: [0u32; VIDEO_WIDTH * VIDEO_HEIGHT],
            flags: [0u8; FLAGS_SIZE],
            delay_timer: 0,
            register: [0u8; REGISTER_SIZE],
            sound_timer: 0,
            opcode: "NOP".to_string(),
            sp: 0,
            index: 0,
            stack: [0u16; STACK_SIZE],
            keypad: [false; KEYPAD_SIZE],
            render: true,
        }
    }
}

impl std::fmt::Display for Chip8 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(
            f,
            "{:<32}{}",
            format!("Opcode: {}", self.opcode),
            format_args!("Stack Pointer: {:#04x}", self.sp)
        )?;

        for x in 0..16 {
            writeln!(
                f,
                "{:<32}{}",
                format!("V{i}: {:#04x}", self.register[x], i = x),
                format_args!("Stack[{i}]: {:#04x}", self.stack[x], i = x)
            )?;
        }

        writeln!(
            f,
            "{:<32}{}",
            format!("PC: {:#04x}", self.pc),
            format_args!("I: {:#04x}", self.index)
        )?;

        Ok(())
    }
}

impl Chip8 {
    pub fn load_rom(filepath: PathBuf) -> std::io::Result<Self> {
        let mut chip = Chip8::default();

        let buffer = std::fs::read(filepath)?;

        // Load the ROM contents into the Chip8's memory, starting at 0x200
        for (idx, byte) in buffer.into_iter().enumerate() {
            chip.memory[(START_ADDRESS as usize) + idx] = byte;
        }

        Ok(chip)
    }
    pub fn set_key(&mut self, key: usize, state: bool) {
        self.keypad[key] = state;
    }

    pub fn get_video_buffer(&self) -> &[u32] {
        &self.video
    }

    pub fn cycle(&mut self) {
        let opcode = ((self.memory[self.pc as usize] as u16) << 8)
            | self.memory[(self.pc as usize) + 1] as u16;

        self.format_opcode(opcode);
        self.pc += 2;

        print!("\x1B[1;1H{}", self);

        self.table(opcode);

        if self.delay_timer > 0 {
            self.delay_timer -= 1;
        }

        if self.sound_timer > 0 {
            self.sound_timer -= 1;
        }
    }

    fn format_opcode(&mut self, opcode: u16) {
        self.opcode = match (opcode & 0xF000) >> 12 {
            0x0 => match opcode & 0x000F {
                0x0 => "CLR",
                0xE => "RET",
                _ => "[T0] NOP",
            },
            0x1 => "JMP",
            0x2 => "Call nnn",
            0x3 => "Skp VX == NN",
            0x4 => "Skp VX != NN",
            0x5 => "Skp VX == VY",
            0x6 => "VX = NN",
            0x7 => "VX += NN",
            0x8 => match opcode & 0x000F {
                0x0 => "VX = VY",
                0x1 => "VX |= VY",
                0x2 => "VX &= VY",
                0x3 => "VX ^= VY",
                0x4 => "VX += VY",
                0x5 => "VX -= VY",
                0x6 => "VX >>= 1",
                0x7 => "VX = VY - VX",
                0xE => "VX <<= 1",
                _ => "[T8] NOP",
            },
            0x9 => "Skip if VX != VY",
            0xA => "I = 0xNNN",
            0xB => "Jump to V0 + 0xNNN",
            0xC => "VX = rand() & 0xNN",
            0xD => "Draw sprite at (VX, VY)",
            0xE => match opcode & 0x000F {
                0x1 => "Skp if key isn't pressed",
                0xE => "Skp if key VX is pressed",
                _ => "[TE] NOP",
            },
            0xF => match opcode & 0x00FF {
                0x07 => "VX = Delay Timer",
                0x0A => "Wait for key, store in VX",
                0x15 => "Delay Timer = VX",
                0x18 => "Sound Timer = VX",
                0x1E => "I += VX",
                0x29 => "Set I to addr of font char in VX",
                0x33 => "Stores BCD encoding of VX into I",
                0x55 => "Store V0 thru VX in RAM from I",
                0x65 => "LD",
                _ => "[TF] NOP",
            },
            _ => "NOP",
        }
        .to_string()
    }

    fn table(&mut self, opcode: u16) {
        let part = (opcode & 0xF000) >> 12;
        match part {
            0x0 => match opcode & 0x000F {
                0x0 => self.op_00_e0(),
                0xE => self.op_00_ee(),
                _ => println!("[Table 00] Unknown opcode: {:#04x}", opcode),
            },
            0x1 => self.op_1nnn(opcode),
            0x2 => self.op_2nnn(opcode),
            0x3 => self.op_3xnn(opcode),
            0x4 => self.op_4xnn(opcode),
            0x5 => self.op_5xyo(opcode),
            0x6 => self.op_6nnn(opcode),
            0x7 => self.op_7xnn(opcode),
            0x8 => match opcode & 0x000F {
                0x0 => self.op_8xy0(opcode),
                0x1 => self.op_8xy1(opcode),
                0x2 => self.op_8xy2(opcode),
                0x3 => self.op_8xy3(opcode),
                0x4 => self.op_8xy4(opcode),
                0x5 => self.op_8xy5(opcode),
                0x6 => self.op_8xy6(opcode),
                0x7 => self.op_8xy7(opcode),
                0xE => self.op_8xye(opcode),
                _ => println!("[Table 8] Unknown opcode: {:#04x}", opcode),
            },
            0x9 => self.op_9xyo(opcode),
            0xA => self.op_annn(opcode),
            0xB => self.op_bnnn(opcode),
            0xC => self.op_cxnn(opcode),
            0xD => self.op_dxyn(opcode),
            0xE => match opcode & 0x000F {
                0x1 => self.op_exa1(opcode),
                0xE => self.op_ex9e(opcode),
                _ => println!("[Table E] Unknown opcode: {:#04x}", opcode),
            },
            0xF => match opcode & 0x00FF {
                0x07 => self.op_fx07(opcode),
                0x0A => self.op_fx0a(opcode),
                0x15 => self.op_fx15(opcode),
                0x18 => self.op_fx18(opcode),
                0x1E => self.op_fx1e(opcode),
                0x29 => self.op_fx29(opcode),
                0x33 => self.op_fx33(opcode),
                0x55 => self.op_fx55(opcode),
                0x65 => self.op_fx65(opcode),
                _ => println!("[Table F] Unknown opcode: {:#04x}", opcode),
            },
            _ => println!("Unknown opcode: {:#04x}", opcode),
        }
    }

    fn rand_byte(&mut self) -> u8 {
        self.rng.gen_range(0..255)
    }

    // ++++++++++++++++++++++++++++++++++++++++++++++++
    //
    // Start Op codes handling
    //
    // ++++++++++++++++++++++++++++++++++++++++++++++++

    /// Clear the display.
    fn op_00_e0(&mut self) {
        self.video.fill(0);
        self.render = true;
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
    fn op_1nnn(&mut self, opcode: u16) {
        self.pc = opcode & 0x0FFF;
    }
    /// Call subroutine at nnn.
    ///
    /// When we call a subroutine, we want to return eventually, so we put the current PC onto the top of the stack.
    /// Remember that we did pc += 2 in Cycle(), so the current PC holds the next instruction after this CALL,
    /// which is correct. We don’t want to return to the CALL instruction because it would be an infinite loop of CALLs and RETs.
    fn op_2nnn(&mut self, opcode: u16) {
        self.stack[self.sp as usize] = self.pc;

        self.sp += 1;

        self.pc = opcode & 0x0FFF;
    }

    /// Skip next instruction if Vx = kk.
    ///
    /// if vx != NN then
    fn op_3xnn(&mut self, opcode: u16) {
        let vx = self.register[((opcode & 0x0F00) >> 8) as usize] as u16;
        let byte = opcode & 0x00FF;

        if vx == byte {
            self.pc += 2;
        }
    }
    /// Skip next instruction if Vx != kk.
    ///
    /// if vx == NN then
    fn op_4xnn(&mut self, opcode: u16) {
        let vx = self.register[((opcode & 0x0F00) >> 8) as usize] as u16;
        let byte = opcode & 0x00FF;

        if vx != byte {
            self.pc += 2;
        }
    }
    /// Skip next instruction if Vx = Vy.
    fn op_5xyo(&mut self, opcode: u16) {
        let vx = self.register[((opcode & 0x0F00) >> 8) as usize];
        let vy = self.register[((opcode & 0x00F0) >> 4) as usize];

        if vx == vy {
            self.pc += 2;
        }
    }
    /// Set Vx = kk.
    fn op_6nnn(&mut self, opcode: u16) {
        self.register[((opcode & 0x0F00) >> 8) as usize] = (opcode & 0x00FF) as u8;
    }

    /// Set Vx = Vx + kk.
    fn op_7xnn(&mut self, opcode: u16) {
        let vx = ((opcode & 0x0F00) >> 8) as usize;
        self.register[vx] = self.register[vx].wrapping_add((opcode & 0x00FF) as u8);
    }
    /// Set Vx = Vy.
    fn op_8xy0(&mut self, opcode: u16) {
        self.register[((opcode & 0x0F00) >> 8) as usize] =
            self.register[((opcode & 0x00F0) >> 4) as usize];
    }
    /// Set Vx = Vx OR Vy.
    fn op_8xy1(&mut self, opcode: u16) {
        self.register[((opcode & 0x0F00) >> 8) as usize] |=
            self.register[((opcode & 0x00F0) >> 4) as usize];
    }
    /// Set Vx = Vx AND Vy.
    fn op_8xy2(&mut self, opcode: u16) {
        self.register[((opcode & 0x0F00) >> 8) as usize] &=
            self.register[((opcode & 0x00F0) >> 4) as usize];
    }
    /// Set Vx = Vx XOR Vy.
    fn op_8xy3(&mut self, opcode: u16) {
        self.register[((opcode & 0x0F00) >> 8) as usize] ^=
            self.register[((opcode & 0x00F0) >> 4) as usize];
    }

    /// Set Vx = Vx + Vy, set VF = carry.
    ///
    /// The values of Vx and Vy are added together.
    /// If the result is greater than 8 bits (i.e., > 255,) VF is set to 1, otherwise 0.
    /// Only the lowest 8 bits of the result are kept, and stored in Vx.
    fn op_8xy4(&mut self, opcode: u16) {
        let vx = ((opcode & 0x0F00) >> 8) as usize;
        let vy = self.register[((opcode & 0x00F0) >> 4) as usize];

        let (sum, overflow) = self.register[vx].overflowing_add(vy);

        self.register[0xF] = if overflow { 1 } else { 0 };

        self.register[vx] = sum;
    }

    /// Set Vx = Vx - Vy, set VF = NOT borrow.
    ///
    /// If Vx > Vy, then VF is set to 1, otherwise 0.
    /// Then Vy is subtracted from Vx, and the results stored in Vx.
    fn op_8xy5(&mut self, opcode: u16) {
        let vx = ((opcode & 0x0F00) >> 8) as usize;
        let vy = ((opcode & 0x00F0) >> 4) as usize;

        self.register[0xF] = if self.register[vx] > self.register[vy] {
            1
        } else {
            0
        };

        self.register[vx] = self.register[vx].wrapping_sub(self.register[vy]);
    }

    /// Set Vx = Vx SHR 1.
    ///
    /// If the least-significant bit of Vx is 1, then VF is set to 1, otherwise 0.
    /// Then Vx is divided by 2.
    fn op_8xy6(&mut self, opcode: u16) {
        let vx = ((opcode & 0xF00) >> 8) as usize;

        self.register[0xF] = self.register[vx] & 0x1;
        self.register[vx] >>= 1;
    }

    /// Set Vx = Vy - Vx, set VF = NOT borrow.
    ///
    /// Vy > Vx, then VF is set to 1, otherwise 0. Then Vx is subtracted from Vy,
    /// and the results stored in Vx.
    fn op_8xy7(&mut self, opcode: u16) {
        let vx = ((opcode & 0x0F00) >> 8) as usize;
        let vy = ((opcode & 0x00F0) >> 4) as usize;

        self.register[0xF] = if self.register[vy] > self.register[vx] {
            1
        } else {
            0
        };

        self.register[vx] = self.register[vy].wrapping_sub(self.register[vx]);
    }

    /// Set Vx = Vx SHL 1.
    ///
    /// If the most-significant bit of Vx is 1, then VF is set to 1, otherwise to 0.
    /// Then Vx is multiplied by 2.
    fn op_8xye(&mut self, opcode: u16) {
        let vx = ((opcode & 0x0F00) >> 8) as usize;

        self.register[0xF] = (self.register[vx] & 0x80) >> 7;

        self.register[vx] <<= 1;
    }
    /// Skip next instruction if Vx != Vy.
    fn op_9xyo(&mut self, opcode: u16) {
        let vx = self.register[((opcode & 0x0F00) >> 8) as usize];
        let vy = self.register[((opcode & 0x00F0) >> 4) as usize];

        if vx != vy {
            self.pc += 2;
        }
    }

    /// Set I = nnn.
    fn op_annn(&mut self, opcode: u16) {
        self.index = opcode & 0x0FFF;
    }
    /// Jump to location nnn + V0.
    fn op_bnnn(&mut self, opcode: u16) {
        self.pc = (self.register[0] as u16).wrapping_add(opcode & 0x0FFF);
    }
    /// Set Vx = random byte AND kk.
    fn op_cxnn(&mut self, opcode: u16) {
        let vx = ((opcode & 0x0F00) >> 8) as usize;
        let byte = (opcode & 0x00FF) as u8;

        self.register[vx] = self.rand_byte() & byte;
    }
    /// Display n-byte sprite starting at memory location I at (Vx, Vy), set VF = collision.
    fn op_dxyn(&mut self, opcode: u16) {
        let pos_x = self.register[((opcode & 0x0F00) >> 8) as usize] as usize % VIDEO_WIDTH;
        let pos_y = self.register[((opcode & 0x00F0) >> 4) as usize] as usize % VIDEO_HEIGHT;
        let height = (opcode & 0x000F) as usize;

        self.register[0xF] = 0;

        for row in 0..height {
            let sprite_byte = self.memory[(self.index as usize) + row];

            for col in 0..8 {
                let sprite_pixel = (sprite_byte & (0x80 >> (col as u8))) > 0x00;
                let screen_pixel =
                    self.video[(pos_y + row) * VIDEO_WIDTH + (pos_x + col)].borrow_mut();
                if sprite_pixel {
                    if *screen_pixel == 0xFFFFFFFF {
                        self.register[0xF] = 1;
                    }

                    *screen_pixel ^= 0xFFFFFFFF;
                }
            }
        }

        self.render = true;
    }
    /// Skip next instruction if key with the value of Vx is pressed.
    fn op_ex9e(&mut self, opcode: u16) {
        let key = self.register[((opcode & 0x0F00) >> 8) as usize] as usize;

        if self.keypad[key] {
            self.pc += 2;
        }
    }
    /// Skip next instruction if key with the value of Vx is not pressed.
    fn op_exa1(&mut self, opcode: u16) {
        let key = self.register[((opcode & 0x0F00) >> 8) as usize] as usize;

        if !self.keypad[key] {
            self.pc += 2;
        }
    }
    /// Set Vx = delay timer value.
    fn op_fx07(&mut self, opcode: u16) {
        self.register[((opcode & 0x0F00) >> 8) as usize] = self.delay_timer;
    }

    /// Wait for a key press, store the value of the key in Vx.
    fn op_fx0a(&mut self, opcode: u16) {
        if let Some(idx) = self.keypad.iter().position(|&x| x) {
            self.register[((opcode & 0x0F00) >> 8) as usize] = idx as u8;
        } else {
            self.pc -= 2;
        }
    }

    /// Set delay timer = Vx.
    fn op_fx15(&mut self, opcode: u16) {
        self.delay_timer = self.register[((opcode & 0x0F00) >> 8) as usize];
    }
    /// Set sound timer = Vx.
    fn op_fx18(&mut self, opcode: u16) {
        self.sound_timer = self.register[((opcode & 0x0F00) >> 8) as usize];
    }
    /// Set I = I + Vx.
    fn op_fx1e(&mut self, opcode: u16) {
        self.index = self
            .index
            .wrapping_add(self.register[((opcode & 0x0F00) >> 8) as usize] as u16);
    }

    /// Set I = location of sprite for digit Vx.
    fn op_fx29(&mut self, opcode: u16) {
        let digit = self.register[((opcode & 0x0F00) >> 8) as usize] as u16;

        self.index = (FONTSET_START_ADDRESS as u16) + (5 * digit);
    }

    /// Store BCD representation of Vx in memory locations I, I+1, and I+2.
    ///
    /// The interpreter takes the decimal value of Vx, and places the hundreds digit in memory at location in I,
    /// the tens digit at location I+1, and the ones digit at location I+2.
    fn op_fx33(&mut self, opcode: u16) {
        let vx = self.register[((opcode & 0x0F00) >> 8) as usize];
        let ui = self.index as usize;
        self.memory[ui] = vx / 100;
        self.memory[ui + 1] = (vx / 10) % 10;
        self.memory[ui + 2] = (vx % 100) % 10;
    }

    /// Store registers V0 through Vx in memory starting at location I.
    fn op_fx55(&mut self, opcode: u16) {
        let vx = ((opcode & 0x0F00) >> 8) as usize;

        for i in 0..=vx {
            self.memory[(self.index as usize) + i] = self.register[i];
        }
    }

    /// Read registers V0 through Vx from memory starting at location I.
    fn op_fx65(&mut self, opcode: u16) {
        let vx = ((opcode & 0x0F00) >> 8) as usize;

        for i in 0..=vx {
            self.register[i] = self.memory[(self.index as usize) + i];
        }
    }

    // Octo Extensions

    /// save an inclusive range of registers to memory starting at i
    fn op_5xy2(&mut self, opcode: u16) {}
    /// load an inclusive range of registers from memory starting at i.
    fn op_5xy3(&mut self, opcode: u16) {}
    /// save v0-vn to flag registers. (generalizing SCHIP).
    fn op_fn75(&mut self, opcode: u16) {}
    /// restore v0-vn from flag registers. (generalizing SCHIP).
    fn op_fn85(&mut self, opcode: u16) {}

    /// load i with a 16-bit address.
    ///
    /// i := long NNNN (0xF000, 0xNNNN)
    fn op_f000(&mut self, opcode: u16) {
        // fn op_nnnn(&mut self, opcode: u16){}
    }
    /// select zero or more drawing planes by bitmask (0 <= n <= 3).
    fn op_fn01(&mut self, opcode: u16) {}
    /// store 16 bytes starting at i in the audio pattern buffer.
    fn op_fnn2(&mut self, opcode: u16) {}
    /// set the audio pattern playback rate to 4000*2^((vx-64)/48)Hz.
    fn op_fx3a(&mut self, opcode: u16) {}
    /// scroll up the contents of the display up by 0-15 pixels.
    fn op_00dn(&mut self, opcode: u16) {}

    // SuperChip8 Extention

    /// Scroll display N lines down
    fn op_00cn(&mut self, opcode: u16) {}

    /// Scroll display 4 pixels right
    fn op_00fb(&mut self) {}
    /// Scroll display 4 pixels left
    fn op_00fc(&mut self) {}
}

#[cfg(test)]
mod tests {
    use super::*;

    fn init_chip() -> Chip8 {
        Chip8::default()
    }

    #[test]
    fn test_op_00_e0() {
        let mut chip = init_chip();
        chip.video[2] = 6;
        chip.video[4] = 5;

        chip.op_00_e0();

        assert_eq!(chip.video, *vec![0; VIDEO_WIDTH * VIDEO_HEIGHT]);
    }

    #[test]
    fn test_op_00_ee() {
        let mut chip = init_chip();
        chip.stack[0] = 512;
        chip.sp = 1;

        chip.op_00_ee();

        assert_eq!(chip.pc, 512);
    }

    #[test]
    fn test_op_1nnn() {
        let mut chip = init_chip();

        chip.op_1nnn(0x1FFF);

        assert_eq!(chip.pc, 0xFFF);
    }

    #[test]
    fn test_op_2nnn() {
        let mut chip = init_chip();

        chip.op_2nnn(0x2FFF);

        assert_eq!(chip.stack[0], START_ADDRESS);
        assert_eq!(chip.sp, 1);
        assert_eq!(chip.pc, 0xFFF);
    }

    #[test]
    fn test_op_3xnn() {
        let mut chip = init_chip();

        chip.op_3xnn(0x30FF);

        assert_eq!(chip.pc, START_ADDRESS);

        chip.register[0] = 0xFF;
        chip.op_3xnn(0x30FF);

        assert_eq!(chip.pc, START_ADDRESS + 2);
    }

    #[test]
    fn test_op_4xnn() {
        let mut chip = init_chip();

        chip.register[0] = 0xFF;
        chip.op_4xnn(0x40FF);

        assert_eq!(chip.pc, START_ADDRESS);

        chip.register[0] = 1;
        chip.op_4xnn(0x40FF);
        assert_eq!(chip.pc, START_ADDRESS + 2)
    }

    #[test]
    fn test_op_5xyo() {
        let mut chip = init_chip();

        chip.register[15] = 1;

        chip.op_5xyo(0x5FF0);

        assert_eq!(chip.pc, START_ADDRESS + 2);
    }

    #[test]
    fn test_op_6nnn() {
        let mut chip = init_chip();

        // Set register 0 to 255
        chip.op_6nnn(0x60FF);

        assert_eq!(chip.register[0], 255);
    }

    #[test]
    fn test_op_7xnn() {
        let mut chip = init_chip();

        chip.register[0] = 2;

        // add 16 to register 0
        chip.op_7xnn(0x7010);

        assert_eq!(chip.register[0], 18);

        // Test overflow

        chip.register[0] = 1;

        chip.op_7xnn(0x70FF);

        assert_eq!(chip.register[0], 0);
    }

    #[test]
    fn test_op_8xy0() {
        let mut chip = init_chip();

        chip.register[0] = 2;
        // assigen register 1 value to register 0
        chip.op_8xy0(0x8100);

        assert_eq!(chip.register[0], chip.register[1])
    }

    #[test]
    fn test_op_8xy1() {
        let mut chip = init_chip();

        chip.register[0] = 0xA;
        chip.register[1] = 0xB;

        // A | B

        // 1010 1011
        // 1011

        // set register 0 to register 0 or register 1
        chip.op_8xy1(0x8011);

        assert_eq!(chip.register[0], 0xB);
    }

    #[test]
    fn test_op_8xy2() {
        let mut chip = init_chip();

        chip.register[0] = 0x03; // 0011
        chip.register[1] = 0x04; // 0100
                                 // 0000

        // set register 0 to register 0 and register 1
        chip.op_8xy2(0x8012);

        assert_eq!(chip.register[0], 0);
    }

    #[test]
    fn test_op_8xy3() {
        let mut chip = init_chip();

        chip.register[0] = 0x03; // 0011
        chip.register[1] = 0x04; // 0100
                                 // 0111

        // set register 0 to register 0 and register 1
        chip.op_8xy3(0x8013);

        assert_eq!(chip.register[0], 7);
    }

    #[test]
    fn test_op_8xy4() {
        let mut chip = init_chip();

        chip.register[0] = 0xFF;
        chip.register[1] = 0xFF;

        // add register 1 to register 0 and set carry if needed
        chip.op_8xy4(0x8014);

        assert_eq!(chip.register[0], 0xFE);
        assert_eq!(chip.register[0xF], 1);
    }

    #[test]
    fn test_op_8xy5() {
        let mut chip = init_chip();

        // Test underflow

        chip.register[0] = 1;
        chip.register[1] = 4;

        chip.op_8xy5(0x8015);

        assert_eq!(chip.register[0], 253);
        assert_eq!(chip.register[1], 4);
        assert_eq!(chip.register[0xF], 0);

        // Non underflow

        chip.register[0] = 4;
        chip.register[1] = 1;

        chip.op_8xy5(0x8015);

        assert_eq!(chip.register[0], 3);
        assert_eq!(chip.register[1], 1);
        assert_eq!(chip.register[0xF], 1);
    }

    #[test]
    fn test_op_8xy6() {
        let mut chip = init_chip();

        chip.register[0] = 0xF;

        // right shift register 0's value by 1
        // save dropped bit in register 0xF
        // y bit is unused in this operation
        chip.op_8xy6(0x8006);

        assert_eq!(chip.register[0], 7);
        assert_eq!(chip.register[0xF], 1);

        chip.register[0xF] = 0; // reset
        chip.register[0] = 6;
        chip.op_8xy6(0x8006);

        assert_eq!(chip.register[0], 3);
        assert_eq!(chip.register[0xF], 0);
    }

    #[test]
    fn test_op_8xy7() {
        let mut chip = init_chip();

        // overflowing

        chip.register[0] = 1;
        chip.register[1] = 3;

        // 3 - 1
        //  register 1 sub register 0 and store in register 0,
        // if not overflow set 0xF to 1
        chip.op_8xy7(0x8017);

        assert_eq!(chip.register[0], 2);
        assert_eq!(chip.register[1], 3);
        assert_eq!(chip.register[0xF], 1);

        // non overflow

        chip.register[0xF] = 0;
        chip.register[0] = 3;
        chip.register[1] = 1;

        chip.op_8xy7(0x8017);

        assert_eq!(chip.register[0], 254);
        assert_eq!(chip.register[1], 1);
        assert_eq!(chip.register[0xF], 0);
    }

    #[test]
    fn test_op_8xye() {
        let mut chip = init_chip();

        chip.register[0] = 0xC0;

        chip.op_8xye(0x700E);

        assert_eq!(chip.register[0], 0x80);
        assert_eq!(chip.register[0xF], 1);

        chip.register[0] = 2;
        chip.op_8xye(0x700E);
        assert_eq!(chip.register[0], 4);
        assert_eq!(chip.register[0xF], 0);
    }

    #[test]
    fn test_op_9xyo() {
        let mut chip = init_chip();

        chip.register[0] = 1;
        chip.register[1] = 2;

        chip.op_9xyo(0x9010);

        assert_eq!(chip.pc, START_ADDRESS + 2);
    }
    #[test]
    fn test_op_annn() {
        let mut chip = init_chip();

        chip.op_annn(0xAFFF);

        assert_eq!(chip.index, 0xFFF);
    }
    #[test]
    fn test_op_bnnn() {
        let mut chip = init_chip();
        chip.register[0] = 1;
        chip.op_bnnn(0xBFFE);

        assert_eq!(chip.pc, 0xFFF);
    }

    #[test]
    fn test_op_cxnn() {
        let mut chip = init_chip();
        chip.op_cxnn(0xC0AC);

        println!("Random Byte: {}", chip.register[0]);
    }

    #[test]
    fn test_op_dxyn() {
        let mut chip = init_chip();

        chip.memory[0] = 0xFF;
        chip.memory[1] = 0xFF;

        chip.op_dxyn(0xD002);

        assert_eq!(chip.video[0..8], vec![0xFFFFFFFF; 8]);
        assert_eq!(chip.video[64..72], vec![0xFFFFFFFF; 8]);
    }

    #[test]
    fn test_op_ex9e() {
        let mut chip = init_chip();

        chip.register[0] = 0;
        chip.keypad[0] = true;

        chip.op_ex9e(0xE09E);

        assert_eq!(chip.pc, START_ADDRESS + 2);
    }

    #[test]
    fn test_op_exa1() {
        let mut chip = init_chip();

        chip.register[0] = 0;

        chip.op_exa1(0xE0A1);

        assert_eq!(chip.pc, START_ADDRESS + 2);
    }

    #[test]
    fn test_op_fx07() {
        let mut chip = init_chip();

        chip.delay_timer = 8;
        chip.op_fx07(0xF007);

        assert_eq!(chip.register[0], 8);
    }

    #[test]
    fn test_op_fx0a() {
        let mut chip = init_chip();

        // key is not pressed

        chip.op_fx0a(0xF00A);

        assert_eq!(chip.pc, START_ADDRESS - 2);

        // key is pressed
        chip.pc = START_ADDRESS;
        chip.keypad[1] = true;
        chip.op_fx0a(0xF00A);

        assert_eq!(chip.register[0], 1);
        assert_eq!(chip.pc, START_ADDRESS);
    }

    #[test]
    fn test_op_fx15() {
        let mut chip = init_chip();

        chip.register[0] = 4;
        chip.op_fx15(0xF015);

        assert_eq!(chip.delay_timer, 4);
    }

    #[test]
    fn test_op_fx18() {
        let mut chip = init_chip();

        chip.register[0] = 4;
        chip.op_fx18(0xF018);

        assert_eq!(chip.sound_timer, 4);
    }

    #[test]
    fn test_op_fx1e() {
        let mut chip = init_chip();

        chip.register[0] = 1;
        chip.op_fx1e(0xF01E);

        assert_eq!(chip.index, 1);
    }

    #[test]
    fn test_op_fx29() {
        let mut chip = init_chip();

        chip.register[0] = 4;
        chip.op_fx29(0xF029);

        assert_eq!(chip.index, (FONTSET_START_ADDRESS + 20) as u16)
    }

    #[test]
    fn test_op_fx33() {
        let mut chip = init_chip();

        chip.register[0] = 111;
        chip.op_fx33(0xF033);

        assert_eq!(chip.memory[0], 1);
        assert_eq!(chip.memory[1], 1);
        assert_eq!(chip.memory[2], 1);
    }

    #[test]
    fn test_op_fx55() {
        let mut chip = init_chip();

        chip.register[0] = 3;
        chip.register[1] = 4;
        chip.register[2] = 5;

        // store value from register 0 to register 2 in memory
        chip.op_fx55(0xF255);

        assert_eq!(chip.memory[0], 3);
        assert_eq!(chip.memory[1], 4);
        assert_eq!(chip.memory[2], 5);
    }

    #[test]
    fn test_op_fx65() {
        let mut chip = init_chip();

        chip.memory[0] = 3;
        chip.memory[1] = 4;
        chip.memory[2] = 5;

        // read memory from i to i+2 and store in register 0-2
        chip.op_fx65(0xF265);

        assert_eq!(chip.register[0], 3);
        assert_eq!(chip.register[1], 4);
        assert_eq!(chip.register[2], 5);
    }

    #[test]
    fn test_array_mut() {
        let mut a = [0u32; 5];

        let b = a[2].borrow_mut();

        *b = 4;

        assert_eq!(a[2], 4);
    }

    #[test]
    fn test_draw_font() {
        let mut chip = init_chip();

        chip.register[0] = 0xF; // F

        chip.register[6] = 0; // x and y  pos 0

        chip.op_fx29(0xF026);

        assert_eq!(chip.index, 155);

        chip.op_dxyn(0xD555);

        println!("{:?}", chip.video);
    }
}
