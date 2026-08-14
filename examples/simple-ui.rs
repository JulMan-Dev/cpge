use cpge::gl;
use cpge::gl::context::PlatformContext;

fn main() {
    gl::boot_gl(async || {
        let context = gl::context();
        let mut events = context.events();

        loop {
            let event = events.recv().await.unwrap();
            println!("{:?}", event);
        }
    });

    println!("done")
}
