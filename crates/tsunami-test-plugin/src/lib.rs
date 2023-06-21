use tsunami::Plugin;
use tsunami::Result;

#[derive(Default, Debug)]
struct TestPlugin;

impl Plugin for TestPlugin {
    fn name(&self) -> &'static str {
        "test"
    }

    fn on_load(&mut self) -> Result<()> {
        println!("from plugin");

        Ok(())
    }
}

tsunami::declare_plugin!(TestPlugin, TestPlugin::default);
