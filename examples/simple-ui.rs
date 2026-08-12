use cpge::gl::boot_gl;
use cpge::gl::event::Events;

fn main() {
    boot_gl(async || {
        let mut events = Events::context();

        loop {
            let event = events.poll().await;
            println!("{:?}", event);
        }
    });
}
