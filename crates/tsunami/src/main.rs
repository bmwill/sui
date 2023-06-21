use tsunami::PluginManager;

fn main() {
    println!("hello world");

    let args = std::env::args().collect::<Vec<_>>();

    let mut manager = PluginManager::new();
    manager.load_plugin(&args[1]).unwrap();
}
