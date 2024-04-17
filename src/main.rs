mod chip;
mod platform;
use chip::{Chip8, VIDEO_HEIGHT, VIDEO_WIDTH};
use platform::SDLRenderer;
use std::{mem::size_of, path::PathBuf, str::FromStr, time::Duration};

fn parse_args() -> (u32, Duration, PathBuf) {
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

    (video_scale, cycle_delay, file_path)
}

fn main() {
    let (video_scale, cycle_delay, file_path) = parse_args();
    print!("\x1B[2J\x1B[1;1H\033[?25l");
    let mut platform = SDLRenderer::new(
        (VIDEO_WIDTH as u32) * video_scale,
        (VIDEO_HEIGHT) as u32 * video_scale,
    )
    .expect("Failed to init sdl2");

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
                .render(video_pitch, chip.get_video_buffer())
                .expect("Failed to render.");
        }
    }

    print!("\033[?25h")
}
